# CSE547 Project 

Code for CSE547 Project, 2023

## Requirements
* [PyTorch](http://pytorch.org/)
* [Transformers](https://github.com/huggingface/transformers)
* datasets
* wandb
* tqdm
* scikit-learn

## Dataset

* RVL-CDIP Dataset (See "data/readme.txt" for detailed data descriptions)

## Training and Evaluation

Run the model with the following commands: 

```bash
>> python run.py --task_name "rvl_cdip" --loss "margin"
```

The loss can take "margin" or "self", which means using margin-based or self-supervised contrastive loss respectively.

## SLURM Workload Manager

An example script `model.slurm` is detailed below for submitting a job to the SLURM workload manager of an HPC environment.
Requested resources are detailed in lines prefaced with `#SBATCH`. The bash script below requests 4 a100 gpus with 40 cpus
and 160GB of ram for a job time span of max 8 hours. Modules CUDA and lz4 are loaded for the needed script actions at the
bottom.

```bash
#!/bin/bash

#SBATCH --job-name=wpm-ood
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wrenparismoe@gmail.com

#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpu-freq=Performance
#SBATCH --mem=160G
#SBATCH --gpus=a100:4
#SBATCH --gpu-freq=high
#SBATCH --time=8:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --nice=0

#SBATCH --chdir=/mmfs1/home/parismoe/projects/cse547project
#SBATCH --export=all
#SBATCH --output=proc-out.txt # where STDOUT goes
#SBATCH --error=proc-err.txt # where STDERR goes

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

mkdir -p /tmp/wpm && tar -I lz4 -xf /mmfs1/gscratch/amath/wpm/data.tar.lz4 -C $_
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv-torch2
CUDA_VISIBLE_DEVICES=0,1,2,3 /mmfs1/gscratch/amath/wpm/.venv/venv-torch2/bin/python run.py --num_train_epochs=8 --batch_size=64 --learning_rate=1e-4
```

We can submit the above job to SLURM with the command:
```bash
>> sbatch model.slurm
```

