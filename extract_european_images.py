import os
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor


SCRATCH_DIR = "/disk/scratch/s2089339/my_datasets/osv5m"
TRAIN_DIR = os.path.join(SCRATCH_DIR, "images/train")  
OUTPUT_DIR = os.path.join(SCRATCH_DIR, "images/train_europe")  


if os.path.exists(OUTPUT_DIR):
    print(f"Removing existing train_europe directory: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load image IDs to extract (all European images
image_id_path = os.path.join(SCRATCH_DIR, "europe_image_ids.txt")

if not os.path.exists(image_id_path):
    print(f"ERROR: europe_image_ids.txt not found at {image_id_path}")
    exit(1)

with open(image_id_path, "r") as f:
    europe_image_ids = {line.strip() + ".jpg" for line in f}  # Ensure IDs match filenames

print(f"Extracting {len(europe_image_ids)} European images.")

# Extract images from ZIP files
extracted_count = 0
skipped_count = 0

def extract_images_from_zip(zip_path):
    global extracted_count, skipped_count
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_image_names = zip_ref.namelist()


        print(f"First 10 images inside {os.path.basename(zip_path)}: {zip_image_names[:10]}")


        for image_name in zip_image_names:

            image_id = os.path.basename(image_name).strip()

            if image_id in europe_image_ids:
                zip_ref.extract(image_name, OUTPUT_DIR)
                extracted_count += 1
                if extracted_count % 10000 == 0:
                    print(f"Extracted {extracted_count} images so far...")
            else:
                skipped_count += 1


zip_files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith(".zip")]

if not zip_files:
    print(f"ERROR: No ZIP files found in {TRAIN_DIR}. Exiting.")
    exit(1)

print(f"Found {len(zip_files)} ZIP files. Beginning extraction...")

with ThreadPoolExecutor(max_workers=8) as executor:  # 🔥 Use 8 threads for faster extraction
    executor.map(extract_images_from_zip, zip_files)

print(f"Finished extracting {extracted_count} European images!")
print(f"Skipped {skipped_count} images because they were not in `europe_image_ids.txt`")
