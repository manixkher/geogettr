#!/bin/bash
#SBATCH --job-name=extract_europe_images
#SBATCH --partition=PGR-Standard
#SBATCH --mem=64G                 # Increase memory for faster I/O
#SBATCH --time=12:00:00            # Increase time if needed
#SBATCH --output=logs/extract_europe_images.out
#SBATCH --error=logs/extract_europe_images.err
#SBATCH --cpus-per-task=16         # Use more CPUs for parallel extraction

echo "Starting SLURM job on $(hostname)"

# Activate Python environment
source ~/venvs/geogettr/bin/activate

echo "Using $(nproc) CPU cores"

# Set dataset paths
DATASET_DIR="$HOME/Work/geogettr/my_datasets/osv5m"
SCRATCH_DIR="/disk/scratch/$USER/my_datasets/osv5m"

# Ensure scratch storage directory exists
mkdir -p $SCRATCH_DIR/images

# Move entire train directory to scratch before extracting
echo "Re-transferring dataset to scratch storage..."
rsync -av --progress --delete $DATASET_DIR/images/train/ $SCRATCH_DIR/images/train/

echo "Dataset transfer to scratch complete."

# Move to script directory
cd $HOME/Work/geogettr

# Run the extraction script, which will extract required images directly from ZIPs
python $HOME/Work/geogettr/extract_european_images.py

echo "SLURM job finished extracting European images."
