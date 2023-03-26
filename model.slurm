#!/bin/bash

#SBATCH --job-name=wpm-ood-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wrenparismoe@gmail.com

#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpu-freq=Performance
#SBATCH --mem=100G
#SBATCH --gpus=a100:4
#SBATCH --gpu-freq=high
#SBATCH --time=4:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --nice=0

#SBATCH --chdir=/mmfs1/home/parismoe/projects/cse547project
#SBATCH --export=all
#SBATCH --output=slurm_output/output.txt # where STDOUT goes
#SBATCH --error=slurm_output/errors.txt # where STDERR goes

# Modules to use (optional).
module load cuda/11.8.0
module load cesg/lz4

export MASTER_PORT=12340
export WORLD_SIZE=4

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# or try export MASTER_ADDR = $SLURM_LAUNCH_NODE_IPADDR

mkdir -p /tmp/wpm && cp $PUB/data.tar.lz4 /tmp/wpm && cd /tmp/wpm && tar -I lz4 -xf data.tar.lz4
cd $WS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv-torch2
CUDA_VISIBLE_DEVICES=0,1,2,3 /mmfs1/gscratch/amath/wpm/.venv/venv-torch2/bin/python run.py --num_train_epochs=6 --batch_size=32 --learning_rate=1e-5
