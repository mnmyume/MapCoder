#!/bin/bash
#SBATCH --job-name=trip_planner
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH -o logs/JOB%j.out
#SBATCH -e logs/JOB%j-err.out

#SBATCH --mail-user=endavinci808@gmail.com
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate mapcoder

mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Date: $(date)"

export MODEL="Qwen"
export DATASET="HumanEval"
export STRATEGY="MapCoder"

srun python src/main.py \
    --model $MODEL \
    --dataset $DATASET \
    --strategy $STRATEGY