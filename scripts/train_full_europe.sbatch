#!/bin/bash
#SBATCH --job-name=geogettr_train_full
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:a40:1             # Request 1 GPUs (a40)
#SBATCH --mem=62G                # Request 62GB memory
#SBATCH --time=24:00:00          # Set max runtime (adjust as needed)
#SBATCH --output=logs/train_log_full_europe_experiment.out   # Save log output

echo "Starting SLURM job on $(hostname)"

source ~/venvs/geogettr/bin/activate

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"



# Load modules (if needed)
# module load cuda/11.3  # Change to match your cluster's CUDA version

# Set paths
DATASET_DIR="$HOME/Work/geogettr/my_datasets/osv5m"
SCRATCH_DIR="/disk/scratch/$USER/my_datasets/osv5m"  # Adjust if your cluster uses /tmp or other fast storage
rm -rf $SCRATCH_DIR/images
# Copy dataset to scratch for fast access
mkdir -p $SCRATCH_DIR/images/train_europe
mkdir -p $SCRATCH_DIR/images/test_europe

cp $DATASET_DIR/reduced_train_europe.csv $SCRATCH_DIR/
cp $DATASET_DIR/reduced_test_europe.csv $SCRATCH_DIR/


echo "Copying and extracting train dataset..."
cd $SCRATCH_DIR/images/train_europe
for i in $(seq -w 00 97); do  # Ensures "00" to "97"
    if [ ! -d "$SCRATCH_DIR/images/train_europe/$i" ]; then
        echo "Extracting train images: $i.zip"
        cp -r "$DATASET_DIR/images/train_europe/$i.zip" "$SCRATCH_DIR/images/train_europe"
        unzip -q "$i.zip"
        rm "$i.zip"
        # unzip -q "$SCRATCH_DIR/images/train/$i.zip" -d "$SCRATCH_DIR/images/train/"
        # rm "$SCRATCH_DIR/images/train/$i.zip"
    else
        echo "Train folder $i already exists, skipping extraction."
    fi
done


# Move to training script directory
cd $HOME/Work/geogettr  # Adjust to match your repo location

# Run training (process only /00)
rm -rf ~/.cache/huggingface/datasets/
python train.py --dataset "$SCRATCH_DIR" --epochs 15 --model_name "reduced_europe_experiment"
cd $SCRATCH_DIR
rm -rf my_datasets
echo "SLURM job finished."
