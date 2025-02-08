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

# Load the module
module load cuda/cuda-12.2.2
module load python/python-3.11.4-gcc-12.2.0


# Path
cd ~/Projects/SimilarSampleCrossOmicNMF/

# Activate the environment
source .mogonet.env/bin/activate

# SSH tunneling
ssh HEC-301-14-Server -N -f

# Run the script
srun python main_dim_reduction.py --storage_mode s3 --omics_mode 2omics --disease ov --run_mode integrate --gpu 0