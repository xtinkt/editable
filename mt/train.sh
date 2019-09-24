#!/bin/bash

python3 train.py data-bin/iwslt14.tokenized.de-en  \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 --criterion editable_training_criterion \
    --label-smoothing 0.1 --max-tokens 4096 --tensorboard-logdir train_editable_logs \
    --edit-samples-path edit_iwslt14.tokenized.de-en/bpe_train.txt \
    --save-dir checkpoints_editable \
    --stability-coeff  100 \
    --editability-coeff 100
