#!/bin/bash

# Download and prepare the data
cd fairseq/examples/translation/
bash prepare-iwslt14.sh
cd ../../..

# Preprocess/binarize the data
TEXT=fairseq/examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
