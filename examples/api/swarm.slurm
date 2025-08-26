#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=igen_train
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=450GB
#SBATCH --qos=qos_llm_min
#SBATCH --job-name=swarm_launch_dbg
#SBATCH --exclusive
#SBATCH --time=200:00:00
#SBATCH --output /leonardo_scratch/fast/iGen_train/fdambro1/domyn-swarm/logs/%x.%j.out
#SBATCH --error /leonardo_scratch/fast/iGen_train/fdambro1/domyn-swarm/logs/%x.%j.err
#SBATCH --exclude=lrdn[1261-3456]

# This is your usual Slurm script header, with the necessary options for your job.
source .venv/bin/activate

# And here we run the Python script that launches the swarm.
# Make sure to adjust the path to your script accordingly.
# If you have a specific Python environment, activate it before running the script.
python -u /leonardo_scratch/fast/iGen_train/fdambro1/domyn-swarm/swarm_launch.py
