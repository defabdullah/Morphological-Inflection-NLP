#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/abdullah-morphological-inflection-nlp:latest
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/morphological-inflection-nlp/predict.py' --output_dir '/users/ahmet.susuz/morph/' --test_file '/users/ahmet.susuz/eng_test.txt' --d_model 1024 --num_heads 12 --dff 512 --dropout_rate 0.2
