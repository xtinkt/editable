import math
import random
from copy import copy

import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.data import Dictionary
from fairseq.data.language_pair_dataset import collate
from fairseq.models.transformer import TransformerModel, TransformerDecoder
from fairseq.tasks.translation import TranslationTask

from lib import Editable, BaseEditable, IngraphRMSProp, training_mode, Lambda, copy_and_replace


def encode_fast(voc, line):
    tokens = [voc.indices.get(tok, voc.unk_index) for tok in line.split()]
    tokens.append(voc.eos_index)
    return tokens


def get_sentence_logp(logits, target, padding_ix, mean=True):
    logp = F.log_softmax(logits, dim=-1)
    logp_target_tokens = torch.gather(logp, -1, target[..., None])[..., 0]  # [batch_size, max_len]
    mask = (target != padding_ix).to(dtype=logp.dtype)
    logp_target = (logp_target_tokens * mask).sum(dim=-1)
    if mean:
        logp_target = logp_target / mask.sum(dim=-1)
    return logp_target


def read_edits(data_path : str, src_dict : Dictionary, tgt_dict : Dictionary):
    samples = []
    with open(data_path) as f_in:
        for line in tqdm.tqdm(f_in):
            sentences = line.split('\t')
            src_sent = encode_fast(src_dict, sentences[0])
            target = encode_fast(tgt_dict, sentences[1])
            alternatives = [encode_fast(tgt_dict, x) for x in sentences[2:]]
            samples.append((src_sent, target, alternatives))

    return samples


@register_criterion('editable_training_criterion')
class EditableTrainingCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, args, task: TranslationTask):
        super().__init__(args, task)
        self.task = task
        self.eps = args.label_smoothing
        self.data_path = args.edit_samples_path
        self.editability_coeff = args.editability_coeff
        self.stability_coeff = args.stability_coeff
        self.max_steps = args.edit_max_steps
        self.almost_last = (args.almost_last != 0)
        print('!!!'*30)
        print('Editability coeff:', self.editability_coeff)
        print('Stability coeff:', self.stability_coeff)
        print('Max steps:', self.max_steps)
        print('Edit learning rate:', args.edit_learning_rate)
        print('Almost last:', self.almost_last)
        print('!!!'*30)
        self.optimizer = IngraphRMSProp(learning_rate=args.edit_learning_rate, beta=nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
                                        )
        

        self.samples = read_edits(self.data_path, task.src_dict, task.tgt_dict)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--edit-samples-path', type=str, metavar='D',
                            help='path to training edits tsv')

        parser.add_argument('--stability-coeff', default=1e2, type=float, metavar='D',
                            help='Stability loss multiplier')
        parser.add_argument('--editability-coeff', default=1e2, type=float, metavar='D',
                            help='Failed edit penalty multiplier')
        parser.add_argument('--edit-max-steps', default=10, type=int, metavar='D',
                            help='Max steps to perform during an editing')
        parser.add_argument('--edit-learning-rate', default=1e-3, type=float, metavar='D',
                            help='Learning rate for RMSPror editor')
        parser.add_argument('--almost-last', default=0, type=int, metavar='D',
                            help='if 0  use the last decoder layer to perform an edit else use penultimate')
        # fmt: on

    def get_edited_transformer(self, model, edit_sample, device=None, dtype=torch.int64, **kwargs):
        with torch.no_grad():
            targets = [edit_sample[1], *edit_sample[2]]
            pad_ix = self.task.tgt_dict.pad()
            edit_target = torch.full([len(targets), max(map(len, targets))], fill_value=pad_ix, device=device,
                                     dtype=dtype)
            prev_output_tokens = edit_target.clone()
            for i, seq in enumerate(targets):
                edit_target[i, :len(seq)] = torch.as_tensor(seq, dtype=dtype)
                prev_output_tokens[i, :len(seq)] = torch.as_tensor([self.task.tgt_dict.eos_index] + seq[:-1],
                                                                   dtype=dtype)

            edit_source = torch.as_tensor([edit_sample[0]] * edit_target.shape[0], device=device, dtype=dtype)
            edit_lengths = torch.full(edit_source.shape[:1], len(edit_sample[0]), device=device, dtype=dtype)

        edit_input = {'src_tokens': edit_source,  # [batch, max_src_len]
                      'src_lengths': edit_lengths,  # [batch]
                      'target': edit_target,  # [batch, max_tgt_len]
                      'prev_output_tokens': prev_output_tokens}  # [batch, max_tgt_len]

        while not isinstance(model, TransformerModel):
            model = model.module
        
        if self.almost_last:
            editable_model = EditableTransformer(
                self, model, xbost_fist_layer_id=len(model.decoder.layers) - 2,
                optimizer=self.optimizer, max_steps=self.max_steps,
                get_editable_parameters=lambda decoder_xbost: decoder_xbost.xbost_layers[0].parameters()
            )
        else:
            editable_model = EditableTransformer(
                self, model, xbost_fist_layer_id=len(model.decoder.layers) - 1,
                optimizer=self.optimizer, max_steps=self.max_steps,
                get_editable_parameters=lambda decoder_xbost: decoder_xbost.xbost_layers.parameters()
            )
        with training_mode(model, is_train=False):
            return editable_model.edit(edit_input, **kwargs)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if not model.training:
            return super().forward(model, sample, reduce)

        device = sample['net_input']['src_tokens'].device
        dtype = sample['net_input']['src_tokens'].dtype

        edit_sample = random.choice(self.samples)

        edited_model, success, editability_loss, edit_complexity = \
            self.get_edited_transformer(model, edit_sample, device, dtype)

        net_output = model(**sample['net_input'])

        with training_mode(model, is_train=False):
            edited_output = edited_model(**sample['net_input'])

        main_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ref_logits = net_output[0].detach()
        stability_loss = (F.softmax(ref_logits, dim=-1)
                       * (F.log_softmax(ref_logits, dim=-1) - F.log_softmax(edited_output[0], dim=-1))
                         ).sum(-1).mean()
        loss = main_loss + self.stability_coeff * stability_loss + self.editability_coeff * editability_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'main_loss': utils.item(main_loss.data) if reduce else main_loss.data,
            'editability_loss': utils.item(editability_loss.data) if reduce else editability_loss.data,
            'stability_loss': utils.item(stability_loss.data) if reduce else stability_loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'edit_complexity': edit_complexity
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        xent_outputs_dict = LabelSmoothedCrossEntropyCriterion.aggregate_logging_outputs(logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        if 'editability_loss' not in logging_outputs[0]:
            return xent_outputs_dict

        xent_outputs_dict['editability_loss'] = sum(log['editability_loss'] for log in logging_outputs) / len(
            logging_outputs)
        xent_outputs_dict['main_loss'] = sum(
            log.get('main_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.
        xent_outputs_dict['stability_loss'] = sum(log['stability_loss'] for log in logging_outputs) / len(
            logging_outputs)
        xent_outputs_dict['edit_complexity'] = sum(log['edit_complexity'] for log in logging_outputs) / len(
            logging_outputs)

        return xent_outputs_dict


def loss_function(net_output, target, criterion, transformer, **loss_kwargs):
    """ Compute editability loss. Only apply layers that were not pre-computed in def edit """
    _, nll_losses = criterion.compute_loss(transformer, net_output, dict(target=target), reduce=False)
    logps = get_sentence_logp(net_output[0], target, padding_ix=1, mean=True)
    loss = torch.relu(torch.max(logps[1:]) - logps[0])

    return loss  # scalar


class EditableTransformer(BaseEditable):
    def __init__(self, criterion, transformer: TransformerModel, xbost_fist_layer_id, mean_logp=True, **kwargs):
        super().__init__()
        self.criterion = criterion
        self.transformer = transformer
        self.mean_logp = mean_logp
        self.padding_ix = self.criterion.task.tgt_dict.pad()
        self.xbost_fist_layer_id = xbost_fist_layer_id

        self.editable_xbost = Editable(
            self.TransformerDecoderXBost(transformer.decoder, first_layer_id=xbost_fist_layer_id),
            loss_function=loss_function,
            **kwargs
        )

    def edit(self, edit_input, **kwargs):
        transformer = self.transformer
        assert isinstance(transformer, TransformerModel)
        encoder_out = transformer.encoder(edit_input['src_tokens'], src_lengths=edit_input['src_lengths'])
        decoder_states_pre_xbost = self.decoder_pre_xbost(edit_input['prev_output_tokens'], encoder_out)
        edit_result = self.editable_xbost.edit(
            dict(edit_input, encoder_out=encoder_out, decoder_states_pre_xbost=decoder_states_pre_xbost),
            targets=edit_input['target'], loss_kwargs=dict(criterion=self.criterion, transformer=self.transformer),
            **kwargs)

        edited_xbost, success, loss, complexity = edit_result

        edited_self = EditableTransformer(self.criterion, self.transformer, self.xbost_fist_layer_id,
                                          mean_logp=self.mean_logp)
        edited_self.training = self.training
        edited_self.editable_xbost = edited_xbost
        return Editable.EditResult(edited_self, success, loss, complexity)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.transformer.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_states_pre_xbost = self.decoder_pre_xbost(prev_output_tokens, encoder_out, **kwargs)
        model_out = self.editable_xbost.module.decoder_post_xbost(encoder_out, decoder_states_pre_xbost)
        return model_out

    def recover_transformer(self):
        original_xbost = self.TransformerDecoderXBost(self.transformer.decoder, first_layer_id=self.xbost_fist_layer_id)
        edited_xbost = self.editable_xbost.module
        assert isinstance(original_xbost, self.TransformerDecoderXBost) and isinstance(edited_xbost, self.TransformerDecoderXBost)

        replacement_dict = {}
        edited_params = dict(edited_xbost.named_parameters())
        for key, param in original_xbost.named_parameters():
            replacement_dict[param] = edited_params[key]

        return copy_and_replace(self.transformer, replace=replacement_dict)

    def decoder_pre_xbost(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        decoder = self.transformer.decoder
        # embed positions
        positions = decoder.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if decoder.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = decoder.embed_scale * decoder.embed_tokens(prev_output_tokens)

        if decoder.project_in_dim is not None:
            x = decoder.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=decoder.dropout, training=decoder.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in decoder.layers[:self.xbost_fist_layer_id]:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=decoder.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)
        return inner_states

    class TransformerDecoderXBost(TransformerDecoder):
        """ Temporary module that applies the second part of transformer decoder """

        def __init__(self, decoder: TransformerDecoder, *, first_layer_id):
            nn.Module.__init__(self)
            self.first_layer_id = first_layer_id
            self.xbost_layers = decoder.layers[first_layer_id:]
            assert isinstance(self.xbost_layers, nn.ModuleList)
            self.layer_norm = decoder.layer_norm
            self.project_out_dim = decoder.project_out_dim
            self.output_layer = decoder.output_layer

        def forward(self, edit_input):
            return self.decoder_post_xbost(**edit_input)

        def decoder_post_xbost(self, encoder_out, decoder_states_pre_xbost, **unused):
            """ Apply final decoder layers after forward_pre_xbost """
            incremental_state = None
            inner_states = list(decoder_states_pre_xbost)
            x = decoder_states_pre_xbost[-1]

            attn = None

            # decoder layers: xbost
            for layer in self.xbost_layers:
                x, attn = layer(
                    x,
                    encoder_out['encoder_out'] if encoder_out is not None else None,
                    encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                )
                inner_states.append(x)

            if self.layer_norm:
                x = self.layer_norm(x)

            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

            if self.project_out_dim is not None:
                x = self.project_out_dim(x)

            x = self.output_layer(x)
            return x, {'attn': attn, 'inner_states': inner_states}
