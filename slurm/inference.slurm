#!/bin/bash
#SBATCH -J Qwen_EMR
#SBATCH -o logs/qwen-%A.out
#SBATCH -e logs/qwen-%A.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH -t 24:00:00
source ~/.bashrc
conda activate emsqa-env
echo "Using Python: $(which python)"
nvidia-smi
srun python inference/run_inference.py \
     --input  data/emr_open.json \
     --output data/qwen2_5b_emr_output.json


