#!/bin/bash
#SBATCH --job-name=mobo
#SBATCH --time=7-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --nodelist=watgpu508,watgpu708,watgpu808,watgpu1008
#SBATCH --nodes=1
#SBATCH --ntasks=1

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

export DATASET="MBPPSubset"
export STRATEGY="MapCoderMAS"
export PASS_AT_K="1"
export TEMPERATURE="0.0"
export LANGUAGE="Python3"

srun python src/mobo.py \
    --dataset $DATASET \
    --strategy $STRATEGY \
    --temperature $TEMPERATURE \
    --pass_at_k $PASS_AT_K \
    --language $LANGUAGE
