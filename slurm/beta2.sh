#!/bin/bash
#SBATCH --job-name=f5-tts-training     # Job name
#SBATCH --partition=gpu                # Specify the partition with 4-GPU nodes (e.g., gpu)
#SBATCH --qos=gpu                      # Request GPU quality of service
#SBATCH --gres=gpu:4                  # Request 4 GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=4                  # Number of tasks (equal to the number of GPUs)
#SBATCH --cpus-per-task=6              # Request 6 CPUs per task (adjust based on node configuration)
#SBATCH --mem-per-cpu=16G             # Request 16 GB of memory per CPU (adjust based on your needs)
#SBATCH --time=80:00:00               # Time limit (adjust as needed)
#SBATCH --output=slurm/out/%x_%j.out   # Output file

module load Anaconda3/2022.05
module load SoX

source activate f5-tts

# Use accelerate config to configure for 4 GPUs, then run:
accelerate launch --mixed_precision=fp16 src/f5_tts/train/train.py --config-name F5TTS_Small_train.yaml model.name=F5TTS_Small_beta2 datasets.batch_size_per_gpu=19200 optim.num_warmup_updates=20000 optim.grad_accumulation_steps=4 optim.epochs=4000

echo "Job's done"



