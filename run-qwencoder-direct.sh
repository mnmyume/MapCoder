#!/bin/bash
#SBATCH --job-name=qc_direct
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

#SBATCH -o logs/JOB%j.out
#SBATCH -e logs/JOB%j-err.out

#SBATCH --mail-user=endavinci808@gmail.com
#SBATCH --mail-type=ALL

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mapcoder

mkdir -p outputs
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Date: $(date)"

export MODEL="QwenCoder"
export DATASET="CC"
export STRATEGY="Direct"
export PASS_AT_K="1"
export TEMPERATURE="0"

srun python src/main.py \
    --model $MODEL \
    --dataset $DATASET \
    --strategy $STRATEGY \
    --pass_at_k $PASS_AT_K \
    --temperature $TEMPERATURE