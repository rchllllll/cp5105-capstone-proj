#!/bin/sh
#SBATCH --job-name=capstone
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -x amdgpu1,amdgpu2,amdgpu3
#SBATCH --partition=long
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e0674485@u.nus.edu

python src/main.py --epochs 100 --num-samples 8192 --batch-size 8 --emb-size 20 --lr 5e-5 --num-workers 8 --output-folder-name adamw --lr-scheduler False 