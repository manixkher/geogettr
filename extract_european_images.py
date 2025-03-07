import os
import zipfile
import shutil

# Paths
data_root = "/home/s2089339/Work/geogettr/my_datasets/osv5m/images/train"
output_root = "/home/s2089339/Work/geogettr/my_datasets/osv5m/images/train_europe"
os.makedirs(output_root, exist_ok=True)

# Load image IDs to extract
image_id_path = "/home/s2089339/Work/geogettr/my_datasets/osv5m/europe_image_ids.txt"
with open(image_id_path, "r") as f:
    europe_image_ids = set(f.read().splitlines())  # Convert to set for fast lookup

# Extract only the necessary images
for zip_file in os.listdir(data_root):
    if zip_file.endswith(".zip"):
        zip_path = os.path.join(data_root, zip_file)
        print(f"Processing {zip_file}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for image_name in zip_ref.namelist():
                image_id = os.path.splitext(os.path.basename(image_name))[0]  # Extract just the `id`
                
                if image_id in europe_image_ids:
                    zip_ref.extract(image_name, output_root)

print("Extracted all European images.")
