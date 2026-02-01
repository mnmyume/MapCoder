#!/bin/bash
#SBATCH --job-name=qwen_direct
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


while
  PORT=$(shuf -n 1 -i 10000-60000)
  netstat -atun | grep -q "$PORT"
do
  continue
done

echo "Selected Port for this Job: $PORT"

echo "Starting udocker execution server..."
udocker run -p $PORT:5000 -e NUM_WORKERS=30 exec-eval-container > logs/udocker_$SLURM_JOB_ID.log 2>&1 &
UDOCKER_PID=$!

echo "Waiting for container to be ready..."
for i in {1..12}; do
    if nc -z localhost $PORT; then
        echo "Container is UP and Ready on port $PORT!"
        break
    fi
    echo "Waiting for container service..."
    sleep 5
done

if ! nc -z localhost $PORT; then
    echo "Error: Container failed to start within 60 seconds."
    kill $UDOCKER_PID
    exit 1
fi

cleanup() {
    echo "Job finished. Killing udocker (PID $UDOCKER_PID)..."
    kill $UDOCKER_PID
}
trap cleanup EXIT

export EXEC_SERVER_URL="http://localhost:$PORT"


export MODEL="Qwen"
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