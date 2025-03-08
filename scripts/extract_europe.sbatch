#!/bin/bash
#SBATCH --job-name=extract_europe_images
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:1           # Request 1 GPU (if needed for later processing)
#SBATCH --mem=64G              # Allocate 64GB memory for faster file I/O
#SBATCH --time=06:00:00        # Set max runtime (adjust as needed)
#SBATCH --output=logs/extract_europe_images.out
#SBATCH --error=logs/extract_europe_images.err
#SBATCH --cpus-per-task=16     # Use 16 CPU cores for parallel file operations

echo "🚀 Starting SLURM job on $(hostname)"
echo "🔧 Activating Python environment..."
source ~/venvs/geogettr/bin/activate

# **Set Paths**
SCRATCH_DIR="/disk/scratch/$USER/my_datasets/osv5m"
HOME_DIR="$HOME/Work/geogettr/my_datasets/osv5m"
TRAIN_DIR="$SCRATCH_DIR/images/train"             # Unzipped dataset in scratch
OUTPUT_SCRATCH="$SCRATCH_DIR/images/train_europe" # Extracted European images stored in scratch first
OUTPUT_HOME="$HOME_DIR/images/train_europe"       # After extraction, move to home
IMAGE_IDS_FILE="$SCRATCH_DIR/europe_image_ids.txt" # European image IDs in scratch

# **🔥 Step 1: Reset Scratch Storage**
echo "🗑️ Deleting existing dataset in scratch (if any)..."
rm -rf "$SCRATCH_DIR"

# **🔥 Step 2: Move dataset and europe_image_ids.txt to scratch**
mkdir -p "$SCRATCH_DIR/images"
echo "📂 Copying dataset (98 ZIPs) to scratch..."
rsync -a --progress "$HOME_DIR/images/train/" "$TRAIN_DIR/"

echo "📄 Copying europe_image_ids.txt to scratch..."
rsync -a --progress "$HOME_DIR/europe_image_ids.txt" "$SCRATCH_DIR/"

# **🔥 Step 3: Delete previous train_europe and create fresh one**
echo "🗑️ Removing previous extracted images in scratch..."
rm -rf "$OUTPUT_SCRATCH"
mkdir -p "$OUTPUT_SCRATCH"

# **🔥 Step 4: Run extraction script (in scratch)**
echo "📂 Extracting European images from dataset..."
python ~/Work/geogettr/extract_european_images.py

# **🔥 Step 5: Move extracted images back to home (parallel transfer)**
echo "🚛 Moving extracted images to home directory..."
rsync -a --info=progress2 "$OUTPUT_SCRATCH/" "$OUTPUT_HOME/"

echo "🎉 SLURM job finished extracting European images!"
