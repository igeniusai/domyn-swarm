#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=account_name
#SBATCH --partition=partition_name
#SBATCH --mem=450GB
#SBATCH --qos=qos_name
#SBATCH --job-name=swarm_launch_dbg
#SBATCH --exclusive
#SBATCH --time=200:00:00
#SBATCH --output /path/to/domyn-swarm/logs/%x.%j.out
#SBATCH --error /path/to/domyn-swarm/logs/%x.%j.err
#SBATCH --exclude=lrdn[1261-3456]

# This is your usual Slurm script header, with the necessary options for your job.
source .venv/bin/activate

# And here we run the Python script that launches the swarm.
# Make sure to adjust the path to your script accordingly.
# If you have a specific Python environment, activate it before running the script.
python -u /path/to/domyn-swarm/swarm_launch.py
