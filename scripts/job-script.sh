#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-user=murphy.tian@mail.utoronto.ca
#SBATCh --mail-type=ALL


module load python/3.10
source ~/ditto-env/bin/activate

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
bash ditto-ba-si.sh cuda:0
