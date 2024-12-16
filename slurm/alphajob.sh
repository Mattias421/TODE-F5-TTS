#!/bin/bash
#SBATCH --partition=gpu-h100,gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64G
#SBATCH --output=%x_%j.out

module load Anaconda3/2022.05
module load SoX

source activate f5-tts

accelerate launch --mixed_precision=fp8 src/f5_tts/train/train.py --config-name F5TTS_Small_train.yaml

echo "Job's done"
