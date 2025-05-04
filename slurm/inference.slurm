#!/bin/bash
#SBATCH -J LLM_Inference
#SBATCH -o Qwen-%A.out
#SBATCH -e Qwen-%A.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH -t 72:00:00
module purge
module load minforge
source activate your_environment
srun python your_code

