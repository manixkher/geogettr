#!/bin/bash
#SBATCH --job-name=geogettr_test_full_exp
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:1            # Request 1 GPU (a40)
#SBATCH --mem=20G                    # Request 62GB memory
#SBATCH --time=2:00:00              # Set max runtime (adjust as needed)
#SBATCH --output=logs/test_transform_final.out   # Save log output

echo "Starting SLURM test job on $(hostname)"

source ~/venvs/geogettr/bin/activate

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set paths
DATASET_DIR="$HOME/Work/geogettr/my_datasets/osv5m"
SCRATCH_DIR="/disk/scratch/$USER/my_datasets/osv5m"  # Adjust if your cluster uses /tmp or other fast storage

# Copy test dataset files to scratch for fast access
mkdir -p $SCRATCH_DIR/images/train_europe
mkdir -p $SCRATCH_DIR/images/test_europe

echo "Copying test and train CSV file..."
cp $DATASET_DIR/reduced_train_europe.csv $SCRATCH_DIR/
cp $DATASET_DIR/reduced_test_europe.csv $SCRATCH_DIR/

# (Optional) If your test images are provided in zipped format, you can extract them here.
# For example:
cd $SCRATCH_DIR/images/test_europe
for i in $(seq -w 00 04); do  
    if [ ! -d "$SCRATCH_DIR/images/test_europe/$i" ]; then
        echo "Extracting test images: $i.zip"
        cp -r "$DATASET_DIR/images/test/$i.zip" "$SCRATCH_DIR/images/test_europe"
        unzip -q "$i.zip"
        rm "$i.zip"
    else
        echo "Test folder $i already exists, skipping extraction."
    fi
done

echo "Preparing test dataset..."

# Move to your project directory (adjust if needed)
cd $HOME/Work/geogettr

# Run the test script.
# If test.py accepts command-line arguments (e.g., for dataset path and model path), pass them accordingly.
rm -rf ~/.cache/huggingface/datasets/
python test.py --dataset "$SCRATCH_DIR" --model_path "epochs/reduced_europe_transform_epoch_15.pth" --label_map "reduced_europe_transform_label_to_index.pkl"

echo "SLURM test job finished."
