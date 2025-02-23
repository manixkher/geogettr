#!/bin/bash
#SBATCH --job-name=geogettr_train_full
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:2             # Request 2 GPUs (2080Ti)
#SBATCH --mem=32G                # Request 32GB memory
#SBATCH --time=48:00:00          # Set max runtime (adjust as needed)
#SBATCH --output=logs/train_log_full.out   # Save log output

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
mkdir -p $SCRATCH_DIR/images/test

cp $DATASET_DIR/train.csv $SCRATCH_DIR/
cp $DATASET_DIR/test.csv $SCRATCH_DIR/


echo "Copying and extracting train dataset..."
cd $SCRATCH_DIR/images/train
for i in $(seq -w 00 97); do  # Ensures "00" to "97"
    if [ ! -d "$SCRATCH_DIR/images/train/$i" ]; then
        echo "Extracting train images: $i.zip"
        cp -r "$DATASET_DIR/images/train/$i.zip" "$SCRATCH_DIR/images/train"
        unzip -q "$i.zip"
        rm "$i.zip"
        # unzip -q "$SCRATCH_DIR/images/train/$i.zip" -d "$SCRATCH_DIR/images/train/"
        # rm "$SCRATCH_DIR/images/train/$i.zip"
    else
        echo "Train folder $i already exists, skipping extraction."
    fi
done

# Function to copy and extract all test images (00.zip to 04.zip)
echo "Copying and extracting test dataset..."
cd $SCRATCH_DIR/images/test
for i in $(seq -w 00 04); do  # Ensures "00" to "04"
    if [ ! -d "$SCRATCH_DIR/images/test/$i" ]; then
        echo "Extracting test images: $i.zip"
        cp -r "$DATASET_DIR/images/test/$i.zip" "$SCRATCH_DIR/images/test"
        unzip -q "$i.zip"
        rm "$i.zip"
        # unzip -q "$SCRATCH_DIR/images/test/$i.zip" -d "$SCRATCH_DIR/images/test/"
        # rm "$SCRATCH_DIR/images/test/$i.zip"
    else
        echo "Test folder $i already exists, skipping extraction."
    fi
done

# Move to training script directory
cd $HOME/Work/geogettr  # Adjust to match your repo location

# Run training (process only /00)
python train.py --dataset "$SCRATCH_DIR" --epochs 20 --model_name "full_set"

echo "SLURM job finished."
