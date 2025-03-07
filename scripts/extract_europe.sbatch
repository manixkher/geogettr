#!/bin/bash
#SBATCH --job-name=extract_europe_images
#SBATCH --partition=PGR-Standard
#SBATCH --mem=32G                 # Allocate 32GB memory
#SBATCH --time=12:00:00            # Set max runtime (adjust as needed)
#SBATCH --output=logs/extract_europe_images.out  # Save log output
#SBATCH --error=logs/extract_europe_images.err   # Save error logs
#SBATCH --cpus-per-task=8          # Use 8 CPU cores for parallel extraction

echo "Starting SLURM job for extracting European images on $(hostname)"

# Activate Python environment
source ~/venvs/geogettr/bin/activate

echo "Using $(nproc) CPU cores"

# Set dataset paths
DATASET_DIR="$HOME/Work/geogettr/my_datasets/osv5m"
SCRATCH_DIR="/disk/scratch/$USER/my_datasets/osv5m"  # Faster access on scratch storage

# Copy dataset to scratch storage for fast extraction
mkdir -p $SCRATCH_DIR/images/train
echo "Copying dataset to scratch storage..."
rsync -av --progress $DATASET_DIR/images/train/*.zip $SCRATCH_DIR/images/train/
rsync -av --progress $DATASET_DIR/train.csv $SCRATCH_DIR/

# Move to scratch directory
cd $SCRATCH_DIR/images/train

echo "Unzipping all ZIP files..."
for zip_file in *.zip; do
    unzip -o -q "$zip_file"  # `-o` forces overwrite
    rm "$zip_file"  # Free up space
done

echo "Dataset extraction complete in scratch storage."

# Move to script directory
cd $HOME/Work/geogettr

# Run the extraction script (which will process files in scratch storage)
python $HOME/Work/geogettr/extract_european_images.py --dataset "$SCRATCH_DIR"

echo "SLURM job finished extracting European images."
