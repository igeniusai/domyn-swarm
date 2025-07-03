#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=igen_train
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=450GB
#SBATCH --qos=qos_llm_min
#SBATCH --job-name=swarm_launch_fede
#SBATCH --time=200:00:00
#SBATCH --output /leonardo_scratch/fast/iGen_train/fdambro1/domyn-swarm/%x.%j.out
#SBATCH --error /leonardo_scratch/fast/iGen_train/fdambro1/domyn-swarm/%x.%j.err
#SBATCH --exclude=lrdn[1261-3456]


source /leonardo_scratch/fast/iGen_train/fdambro1/domyn-swarm/.venv/bin/activate

python -u /leonardo_scratch/fast/iGen_train/fdambro1/domyn-swarm/launch.py

