## Get started
1. Get fairseq submodule: `(git submodule init) && (git submodule update)`
2. Install fairseq: `cd fairseq && pip install --editable .`
3. Prepare iwslt14.de-en: `./prepare_iwslt14_de_en.sh` 
4. Prepare edit samples: run notebook `generate_edit_datasets_samples.ipynb`

## Training
Train Editable Transformer: `./train.sh`

## Evaluating
Evaluate trained model run notebook: `evaluate.ipynb`
