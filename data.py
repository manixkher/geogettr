
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from data_test import OSV5MTest  # Importing OSV5MTest


quadtree_file_path = "reduced_quadtree_10_1000.csv"
quadtree_df = pd.read_csv(quadtree_file_path)


# Convert to PyTorch tensor (lat/lon values only)
quadtree_centroids = torch.tensor(
    quadtree_df[["mean_lat", "mean_lon"]].values, dtype=torch.float32, device="cuda"
)


quadtree_cluster_ids = torch.tensor(
    quadtree_df["cluster_id"].values, dtype=torch.long, device="cuda"
)


def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371):
    """Compute Haversine distance between two sets of points using PyTorch (GPU-accelerated)."""
    # lat1, lon1, lat2, lon2 = map(torch.radians, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = map(lambda x: x * (torch.pi / 180), [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    return earth_radius * c  # Returns distance in km

class OSV5MDataset(Dataset):
    def __init__(self, split="train", transform=None, tau=75, limit=15, val_ratio=0.1, dataset_path = None, label_to_index=None):
        """
        Args:
            split (str): "train" or "test"
            transform: Image transformation pipeline
            tau (float): Temperature parameter for label smoothing
        """
        dataset_builder = OSV5MTest(full=True, dataset_path=dataset_path)
        dataset_builder.download_and_prepare()
        self.dataset = dataset_builder.as_dataset(split=split)
        # self.dataset = self.dataset.select(range(min(len(self.dataset), limit)))  # Limit dataset for testing

        print(f"Loaded {split} dataset! Total of {len(self.dataset)} images")
        self.transform = transform if transform else self.default_transform()
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        # Create a mapping from the original label to a new contiguous index
        self.label_to_index = label_to_index
        print("Label mapping:", self.label_to_index)

    def _haversine_label_smoothing(self, lat, lon):
        """Generate smoothed geocell labels based on haversine distance."""
     

        latlon = torch.tensor([lat, lon], dtype=torch.float32, device=self.device)

        # Compute Haversine distances to all geocells 
        dists = haversine_distance(
            latlon[0].expand(quadtree_centroids.shape[0]),
            latlon[1].expand(quadtree_centroids.shape[0]),
            quadtree_centroids[:, 0],
            quadtree_centroids[:, 1]
        )

        # Find the true geocell (nearest geocell)
        min_dist, true_geocell_idx = torch.min(dists, dim=0)
        true_geocell = quadtree_cluster_ids[true_geocell_idx]

        # Compute label smoothing
        smoothed_probs = torch.exp(-(dists - min_dist) / self.tau)
        # Normalisation
        smoothed_probs /= smoothed_probs.sum()

        return smoothed_probs
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Retrieve an image + smoothed geocell labels."""
        
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB") if isinstance(sample["image"], Image.Image) else sample["image"]
        image = self.transform(image)
        
        lat, lon = sample["latitude"], sample["longitude"]
        geocell = sample["quadtree_10_1000"]
        geocell = self.label_to_index[geocell]
        label = self._haversine_label_smoothing(lat, lon)
        latlon_tensor = torch.tensor([lat, lon], dtype=torch.float32, device=self.device)

        
        return image, label, latlon_tensor, geocell
    
    @staticmethod
    def default_transform():
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

