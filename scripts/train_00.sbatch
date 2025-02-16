#!/bin/bash
#SBATCH --job-name=geogettr_train_00
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:1             # Request 2 GPUs (2080Ti)
#SBATCH --mem=20G                # Request 32GB memory
#SBATCH --time=03:00:00          # Set max runtime (adjust as needed)
#SBATCH --output=logs/train_log_00.out   # Save log output

echo "Starting SLURM job on $(hostname)"

source ~/venvs/geogettr/bin/activate

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"



# Load modules (if needed)
# module load cuda/11.3  # Change to match your cluster's CUDA version

# Set paths
DATASET_DIR="$HOME/Work/geogettr/my_datasets/osv5m"
SCRATCH_DIR="/disk/scratch/$USER/my_datasets/osv5m"  # Adjust if your cluster uses /tmp or other fast storage

# Copy dataset to scratch for fast access
mkdir -p $SCRATCH_DIR/images/train

# Check if '00' folder already exists, if not, copy and extract
if [ ! -d "$SCRATCH_DIR/images/train/00" ]; then
    echo "Folder 00 not found, copying and extracting 00.zip..."
    cp -r $DATASET_DIR/images/train/00.zip $SCRATCH_DIR/images/train
    cp $DATASET_DIR/train.csv $SCRATCH_DIR/
    
    # Unzip dataset
    cd $SCRATCH_DIR/images/train
    unzip 00.zip
    rm 00.zip  # Free up space
else
    echo "Folder 00 already exists, skipping extraction."
fi

# Unzip dataset
cd $SCRATCH_DIR/images/train
unzip 00.zip
rm 00.zip  # Free up space

# Move to training script directory
cd $HOME/Work/geogettr  # Adjust to match your repo location

# Run training (process only /00)
python train.py --dataset "$SCRATCH_DIR" --epochs 5

echo "SLURM job finished."
