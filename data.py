# import os
# import pandas as pd
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# from datasets import load_dataset
# from PIL import Image

# def haversine_distance(lat1, lon1, lat2, lon2):
#     """Calculate the great-circle distance between two points on Earth using the Haversine formula."""
#     R = 6371  # Earth radius in km
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#     c = 2 * np.arcsin(np.sqrt(a))
#     return R * c

# class OSV5MDataset(Dataset):
#     def __init__(self, split="train", transform=None, tau=75):
#         """
#         Args:
#             split (str): "train" or "test"
#             transform: Image transformation pipeline
#             tau (float): Temperature parameter for label smoothing
#         """
#         self.dataset = load_dataset("OSV5M", full=True)[split]
#         self.transform = transform if transform else self.default_transform()
#         self.tau = tau
        
#         # Extract geocell centroids from dataset
#         self.geocell_centroids = self._extract_geocell_centroids()
        
#     def _extract_geocell_centroids(self):
#         """Extract unique geocell centroids from the dataset."""
#         geocells = {}
#         for sample in self.dataset:
#             geocell_id = sample["quadtree_10_5000"]  # Choose a quadtree level
#             if geocell_id not in geocells:
#                 geocells[geocell_id] = (sample["latitude"], sample["longitude"])
#         return geocells

#     def _haversine_label_smoothing(self, lat, lon):
#         """Generate smoothed geocell labels based on haversine distance."""
#         smoothed_labels = {}
#         true_geocell = min(self.geocell_centroids, key=lambda g: haversine_distance(lat, lon, *self.geocell_centroids[g]))
        
#         for geocell, (g_lat, g_lon) in self.geocell_centroids.items():
#             d = haversine_distance(lat, lon, g_lat, g_lon)
#             d_true = haversine_distance(lat, lon, *self.geocell_centroids[true_geocell])
#             smoothed_labels[geocell] = np.exp(-(d - d_true) / self.tau)
        
#         # Normalize
#         total = sum(smoothed_labels.values())
#         for geocell in smoothed_labels:
#             smoothed_labels[geocell] /= total
        
#         return torch.tensor(list(smoothed_labels.values()), dtype=torch.float32)
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         """Retrieve an image + smoothed geocell labels."""
#         sample = self.dataset[idx]
#         image = sample["image"].convert("RGB")
#         image = self.transform(image)
        
#         # Convert lat/lon into smoothed geocell labels
#         label = self._haversine_label_smoothing(sample["latitude"], sample["longitude"])
        
#         return image, label
    
#     @staticmethod
#     def default_transform():
#         return transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from data_test import OSV5MTest  # Importing OSV5MTest

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth using the Haversine formula."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

class OSV5MDataset(Dataset):
    def __init__(self, split="train", transform=None, tau=75, limit = 5):
        """
        Args:
            split (str): "train" or "test"
            transform: Image transformation pipeline
            tau (float): Temperature parameter for label smoothing
        """
        self.dataset = OSV5MTest(full=True).as_dataset(split=split)
        self.dataset = self.dataset.select(range(min(len(self.dataset), limit))) # Load dataset using OSV5MTest
        print("loaded!!!!")
        self.transform = transform if transform else self.default_transform()
        self.tau = tau
        
        # Extract geocell centroids from dataset
        self.geocell_centroids = self._extract_geocell_centroids()
        
    def _extract_geocell_centroids(self):
        """Extract unique geocell centroids from the dataset."""
        geocells = {}
        for sample in self.dataset:
            geocell_id = sample["quadtree_10_5000"]  # Choose a quadtree level
            if geocell_id not in geocells:
                geocells[geocell_id] = (sample["latitude"], sample["longitude"])
        return geocells

    def _haversine_label_smoothing(self, lat, lon):
        """Generate smoothed geocell labels based on haversine distance."""
        smoothed_labels = {}
        true_geocell = min(self.geocell_centroids, key=lambda g: haversine_distance(lat, lon, *self.geocell_centroids[g]))
        
        for geocell, (g_lat, g_lon) in self.geocell_centroids.items():
            d = haversine_distance(lat, lon, g_lat, g_lon)
            d_true = haversine_distance(lat, lon, *self.geocell_centroids[true_geocell])
            smoothed_labels[geocell] = np.exp(-(d - d_true) / self.tau)
        
        # Normalize
        total = sum(smoothed_labels.values())
        for geocell in smoothed_labels:
            smoothed_labels[geocell] /= total
        
        return torch.tensor(list(smoothed_labels.values()), dtype=torch.float32)
    
    def __len__(self):
        return len(self.dataset)
    
    # def __getitem__(self, idx):
    #     """Retrieve an image + smoothed geocell labels."""
    #     sample = self.dataset[idx]
    #     image = Image.open(sample["image"]).convert("RGB")  # Ensure proper image loading
    #     image = self.transform(image)
        
    #     # Convert lat/lon into smoothed geocell labels
    #     label = self._haversine_label_smoothing(sample["latitude"], sample["longitude"])
        
    #     return image, label
    
    def __getitem__(self, idx):
        """Retrieve an image + smoothed geocell labels."""
        sample = self.dataset[idx]
        
        # Image is already a PIL Image (JpegImageFile), so no need to open it again
        image = sample["image"].convert("RGB") if isinstance(sample["image"], Image.Image) else sample["image"]
        image = self.transform(image)
        
        lat, lon = sample["latitude"], sample["longitude"]
        geocell = sample['cell']
        # Convert lat/lon into smoothed geocell labels
        label = self._haversine_label_smoothing(sample["latitude"], sample["longitude"])
        
        return image, label, (lat, lon), geocell

    
    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

