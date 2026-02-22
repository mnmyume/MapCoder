#!/bin/bash
#SBATCH --job-name=surrogate_sweep
#SBATCH --array=0-119%6
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=watgpu508,watgpu708,watgpu808,watgpu1008
#SBATCH --nodes=1
#
#SBATCH -o logs/surrogate_%A_%a.out
#SBATCH -e logs/surrogate_%A_%a-err.out
#
#SBATCH --mail-user=endavinci808@gmail.com
#SBATCH --mail-type=ALL

# ─── Environment ─────────────────────────────────────────────────────────────
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mapcoder

# ─── Ensure directories exist ────────────────────────────────────────────────
mkdir -p logs
mkdir -p surrogate/data/results

# ─── Diagnostics ─────────────────────────────────────────────────────────────
echo "Job Array ID : $SLURM_ARRAY_JOB_ID"
echo "Task ID      : $SLURM_ARRAY_TASK_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Date         : $(date)"

# ─── Run the benchmark for this task ID ──────────────────────────────────────
srun python surrogate/run_benchmark.py \
    --config_file surrogate/data/all_configs.json \
    --task_id $SLURM_ARRAY_TASK_ID \
    --output_dir surrogate/data/results
