import os
import zipfile

# Paths
data_root = "/disk/scratch/s2089339/my_datasets/osv5m/images/train"
output_root = "/disk/scratch/s2089339/my_datasets/osv5m/images/train_europe"
os.makedirs(output_root, exist_ok=True)

# Load image IDs to extract
image_id_path = "/disk/scratch/s2089339/my_datasets/osv5m/europe_image_ids.txt"
with open(image_id_path, "r") as f:
    europe_image_ids = set(f.read().splitlines())  # Convert to set for fast lookup

print(f"Expecting to extract {len(europe_image_ids)} images.")

# Extract only required images without fully unzipping
extracted_count = 0
zip_files = [f for f in os.listdir(data_root) if f.endswith(".zip")]

if not zip_files:
    print("No ZIP files found! Exiting.")
    exit(1)

for zip_file in zip_files:
    zip_path = os.path.join(data_root, zip_file)
    print(f"Processing {zip_file}...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for image_name in zip_ref.namelist():
            image_id = os.path.splitext(os.path.basename(image_name))[0]  # Extract just the `id`

            if f"{image_id}.jpg" in europe_image_ids:  # Extract only missing images
                zip_ref.extract(image_name, output_root)
                extracted_count += 1

                if extracted_count % 10000 == 0:  # Log every 10K extractions
                    print(f"Extracted {extracted_count} images so far...")

print(f"Finished extracting {extracted_count} missing images.")
