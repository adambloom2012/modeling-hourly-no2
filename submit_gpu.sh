#!/bin/bash
#BSUB -n 2
#BSUB -W 24:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -R "select[a30 || a10 || l40 || a100 || h100 ]"
#BSUB -R "rusage[mem=72GB]"
#BSUB -J train_sat
#BSUB -o logs/out.%J
#BSUB -e logs/err.%J

source ~/.bashrc
source /usr/local/usrapps/atmoschem/abloom/venv/bin/activate #pred has extra packages

python satellite_model/training.py --epochs=20 --datadir=/share/atmoschem/abloom/data/eea
