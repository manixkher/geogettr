
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from data_test import OSV5MTest  # Importing OSV5MTest

# Load Quadtree Data (Preload as a Tensor)
quadtree_file_path = "quadtree_10_1000.csv"
quadtree_df = pd.read_csv(quadtree_file_path)
# quadtree_centroids = dict(zip(quadtree_df["cluster_id"], zip(quadtree_df["mean_lat"], quadtree_df["mean_lon"])))
quadtree_centroids = torch.tensor(
    quadtree_df[["mean_lat", "mean_lon"]].values, dtype=torch.float32, device="cuda"
)
quadtree_cluster_ids = torch.tensor(quadtree_df["cluster_id"].values, dtype=torch.long, device="cuda")

# def haversine_distance(lat1, lon1, lat2, lon2):
#     """Calculate the great-circle distance between two points on Earth using the Haversine formula."""
#     R = 6371  # Earth radius in km
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#     c = 2 * np.arcsin(np.sqrt(a))
#     return R * c

def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371):
    """Compute Haversine distance between two sets of points using PyTorch (GPU-accelerated)."""
    lat1, lon1, lat2, lon2 = map(torch.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    return earth_radius * c  # Returns distance in km

class OSV5MDataset(Dataset):
    def __init__(self, split="train", transform=None, tau=75, limit=5, val_ratio=0.1, dataset_path = None):
        """
        Args:
            split (str): "train" or "test"
            transform: Image transformation pipeline
            tau (float): Temperature parameter for label smoothing
        """
        self.dataset = OSV5MTest(full=True, dataset_path=dataset_path).as_dataset(split=split)
        self.dataset = self.dataset.select(range(min(len(self.dataset), limit)))  # Limit dataset for testing

        print(f"Loaded {split} dataset! Total of {len(self.dataset)} images")
        self.transform = transform if transform else self.default_transform()
        self.tau = tau

    def _haversine_label_smoothing(self, lat, lon):
        """Generate smoothed geocell labels based on haversine distance."""
        # smoothed_labels = {}
        # true_geocell = min(quadtree_centroids, key=lambda g: haversine_distance(lat, lon, *quadtree_centroids[g]))
        
        # for geocell, (g_lat, g_lon) in quadtree_centroids.items():
        #     d = haversine_distance(lat, lon, g_lat, g_lon)
        #     d_true = haversine_distance(lat, lon, *quadtree_centroids[true_geocell])
        #     smoothed_labels[geocell] = np.exp(-(d - d_true) / self.tau)
        
        # # Normalize
        # total = sum(smoothed_labels.values())
        # for geocell in smoothed_labels:
        #     smoothed_labels[geocell] /= total
        
        # return torch.tensor(list(smoothed_labels.values()), dtype=torch.float32)

        latlon = torch.tensor([lat, lon], dtype=torch.float32, device=self.device)

        # Compute Haversine distances to all geocells (Vectorized)
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
        label = self._haversine_label_smoothing(lat, lon)
        latlon_tensor = torch.tensor([lat, lon], dtype=torch.float32, device=self.device)

        
        return image, label, latlon_tensor, geocell
    
    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

