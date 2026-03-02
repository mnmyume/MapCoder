#!/bin/bash
#SBATCH --job-name=surrogate_opt
#SBATCH --time=7-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=watgpu508,watgpu708,watgpu808,watgpu1008
#SBATCH --nodes=1
#
#SBATCH -o logs/optimize_%j.out
#SBATCH -e logs/optimize_%j-err.out
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
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Date         : $(date)"
echo "Arguments    : $@"

# ─── Run the optimisation loop ───────────────────────────────────────────────
# This job does NOT need a GPU — it only trains sklearn models and polls squeue.
# Child benchmark jobs (submitted via sbatch inside optimize.py) use GPUs.
srun python surrogate/optimize.py "$@"
