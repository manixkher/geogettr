# "ts 💔"
import os
from my_datasets.osv5m.osv5m_test import OSV5M
import datasets

# Set up paths
DATASET_DIR = "my_datasets/osv5m"
TRAIN_ZIP = os.path.join(DATASET_DIR, "images", "train", "00")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")

# Define a subclass to override _split_generators for single ZIP
class OSV5MTest(OSV5M):
    
    def _generate_examples(self, image_paths, annotation_path):

        df = self.df(annotation_path)
        print(f"DEBUG: Loaded {len(df)} metadata entries")  # ✅ Check if metadata is loaded

        image_paths = list(image_paths)  # Convert generator to list
        print(f"DEBUG: Found {len(image_paths)} images")  # ✅ Check if images are passed

        for idx, image_path in enumerate(image_paths):
            info_id = os.path.splitext(os.path.split(image_path)[-1])[0]  # Extract ID from filename
            print(f"DEBUG: Checking image {image_path}, extracted ID: {info_id}")  # ✅ See extracted ID

            if info_id not in df:
                print(f"⚠️ WARNING: {info_id} not found in metadata!")  # ✅ Warn about missing IDs
                continue

            try:
                example = {
                    "image": image_path,
                }
                example.update(df[info_id])  # ✅ Use update() instead of |
            except Exception as e:
                print(f"❌ ERROR: {e} for ID {info_id}")  # ✅ Catch specific errors
                continue

            yield idx, example


    def _split_generators(self, dl_manager):
        print("HELLO(!)")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_paths": dl_manager.iter_files("my_datasets/osv5m/images/train/00/"),
                    "annotation_path": TRAIN_CSV,
                },
            )
        ]

# Instantiate and prepare the dataset
dataset = OSV5MTest(full=True)
dataset.download_and_prepare()
# print(dataset.as_dataset())
ds = dataset.as_dataset(split="train")

# # Print the first few entries
for i in range(5):  # Print first 5 entries
    print(ds[i])
