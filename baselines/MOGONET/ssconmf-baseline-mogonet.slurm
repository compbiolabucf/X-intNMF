#!/bin/bash
#SBATCH --account=wzhang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --error=log-mogonet.err
#SBATCH --output=log-mogonet.out
#SBATCH --job-name=ariel
#SBATCH --gres=gpu:2


# Path
cd ~/Projects/SimilarSampleCrossOmicNMF/baselines/MOGONET/

# Activate the environment
source .mogonet.env/bin/activate

# SSH tunneling
ssh HEC-301-14-Server -N -f

# Run the script
srun python main_mogonet_custom.py --storage_mode s3 --omics_mode 3omics --disease brca --parallel True