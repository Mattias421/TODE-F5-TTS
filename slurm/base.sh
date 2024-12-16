#!/bin/bash
#SBATCH --job-name=f5-tts-training     # Job name
#SBATCH --partition=gpu-h100,gpu        # Partition: Specify both, Slurm will find a suitable node
#SBATCH --qos=gpu                      # QOS: Request GPU access
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=2                  # Number of tasks (equal to number of GPUs)
#SBATCH --cpus-per-task=8              # Request 8 CPUs per task (2*8 = 16 virtual cores total)
#SBATCH --mem-per-cpu=16G             # Request 16 GB per CPU (16*8 = 128 GB total)
#SBATCH --time=50:00:00               # Time limit (adjust as needed)
#SBATCH --output=slurm/out/%x_%j.out   # Output file

module load Anaconda3/2022.05
module load SoX

source activate f5-tts

# Use accelerate config to configure for 2 GPUs, then run:
accelerate launch --mixed_precision=fp16 src/f5_tts/train/train.py --config-name F5TTS_Base_train.yaml

echo "Job's done"
