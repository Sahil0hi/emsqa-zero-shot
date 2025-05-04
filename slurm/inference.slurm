#!/bin/bash
#SBATCH -J Qwen_EMR
#SBATCH -o qwen-%A.out
#SBATCH -e qwen-%A.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH -t 24:00:00
module purge
module load minforge
source activate your_env
srun python inference/run_inference.py \
     --input  data/emr_open.json \
     --output inference/qwen2_5b_emr_output.json
