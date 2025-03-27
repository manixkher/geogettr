# "ts 💔"
import os
from my_datasets.osv5m.osv5m_test import OSV5M
import datasets

class OSV5MTest(OSV5M):
    def __init__(self, dataset_path, *args, **kwargs):
        self.full = kwargs.pop('full', True)
        print(f"DEBUG: OSV5MTest initialized with full={self.full}")
        self.DATASET_DIR = dataset_path
        self.TRAIN_DIR = os.path.join(self.DATASET_DIR, "images", "train_europe")
        self.TRAIN_CSV = os.path.join(self.DATASET_DIR, "reduced_train_europe.csv")
        self.TEST_DIR = os.path.join(self.DATASET_DIR, "images", "test_europe")
        self.TEST_CSV = os.path.join(self.DATASET_DIR, "reduced_test_europe.csv")
        super().__init__(*args, **kwargs)
        self.full = True
    
    def _generate_examples(self, image_paths, annotation_path):

        df = self.df(annotation_path)
        print(f"DEBUG: Loaded {len(df)} metadata entries") 
        
        image_paths = list(image_paths)  # Convert generator to list
        print(f"DEBUG: Found {len(image_paths)} images")  

        for idx, image_path in enumerate(image_paths):
            info_id = os.path.splitext(os.path.split(image_path)[-1])[0]  # Extract ID from filename
            print(f"DEBUG: Checking image {image_path}, extracted ID: {info_id}") 

            if info_id not in df:
                print(f"WARNING: {info_id} not found in metadata!")
                continue

            try:
                example = {
                    "image": image_path,
                }
                example.update(df[info_id]) 
            except Exception as e:
                print(f"ERROR: {e} for ID {info_id}") 
                continue

            yield idx, example


    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_paths": dl_manager.iter_files(self.TRAIN_DIR),
                    "annotation_path": self.TRAIN_CSV,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_paths": dl_manager.iter_files(self.TEST_DIR),
                    "annotation_path": self.TEST_CSV,
                },
            )
        ]

