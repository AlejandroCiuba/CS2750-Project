#!/bin/env bash

source activate data-analysis  # NOTE: Change this to whatever conda environment you want, or remove it

python splits.py \
    -d data/SPC/SPC.json \
    -J True \
    -f 5 \
    -c fold \
    -r 1000 \
    -s data/SPC-FOLD/SPC.json \
    -S True

echo "DONE"
