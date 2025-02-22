import torch
import torch.nn as nn
import numpy as np
from data import quadtree_centroids
# NOTES TO DO --
# USE GEOCELL CENTROIDS FROM QUADTREE_1000 TO CALCULATE HAVERSINE LOSS. RIGHT NOW IT DOESN'T MAKE SENSE
def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371):
    """Computes the Haversine distance between two lat/lon points."""
    
    # lat1, lon1 = np.radians(coord1)
    # lat2, lon2 = np.radians(coord2)

    # dlat = lat2 - lat1
    # dlon = lon2 - lon1

    # a = (np.sin(dlat / 2) ** 2) + (np.cos(lat1) * np.cos(lat2)) * (np.sin(dlon / 2) ** 2)
    # c = 2 * np.arcsin(np.sqrt(a))

    """Computes the Haversine distance between two lat/lon points using PyTorch (GPU-accelerated)."""

    # lat1, lon1, lat2, lon2 = map(torch.radians, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = map(lambda x: x * (torch.pi / 180), [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    return earth_radius * c  # Returns distance in km

# class HaversineSmoothedLoss(nn.Module):
#     def __init__(self, tau=75.0):
#         super(HaversineSmoothedLoss, self).__init__()
#         self.tau = tau

#         # Convert quadtree centroids to a single GPU tensor for efficiency
#         global quadtree_centroids_tensor
#         quadtree_centroids_tensor = torch.tensor(quadtree_centroids, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

#     # Forward function without using NumPy operations or for loops
#     def forward(self, pred_logits, latlon, geocell_indices):
#         """
#         Compute the Haversine-smoothed classification loss.

#         Args:
#             pred_logits (Tensor): Predicted logits of shape (batch_size, num_classes).
#             latlon (Tensor): Tensor of shape (batch_size, 2) with (latitude, longitude).
#             geocell_indices (Tensor): Tensor of shape (batch_size,) containing the true geocell IDs.

#         Returns:
#             Tensor: Loss value.
#         """
#         batch_size, num_classes = pred_logits.shape
#         device = pred_logits.device  # Ensure computations stay on the correct device

#         # Ensure latlon is a tensor on the correct device
#         latlon = torch.tensor(latlon, dtype=torch.float32, device=device) if isinstance(latlon, list) else latlon

#         # Get true geocell coordinates for each batch sample
#         true_geocell_coords = quadtree_centroids_tensor[geocell_indices]  # Shape: (batch_size, 2)

#         # Compute Haversine distances between each sample and ALL geocells (vectorized)
#         lat1, lon1 = latlon[:, 0].unsqueeze(1), latlon[:, 1].unsqueeze(1)  # Shape: (batch_size, 1)
#         lat2, lon2 = quadtree_centroids_tensor[:, 0], quadtree_centroids_tensor[:, 1]  # Shape: (num_classes,)

#         dists = haversine_distance(lat1, lon1, lat2, lon2)  # Shape: (batch_size, num_classes)

#         # Get distance to the true geocell (vectorized)
#         d_true = haversine_distance(latlon[:, 0], latlon[:, 1], true_geocell_coords[:, 0], true_geocell_coords[:, 1])  # Shape: (batch_size,)

#         # Compute smoothed probabilities
#         smoothed_probs = torch.exp(-(dists - d_true.unsqueeze(1)) / self.tau)  # Shape: (batch_size, num_classes)
#         # Normalisation
#         smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=1, keepdim=True) 

#         # Compute softmax predictions
#         pred_probs = torch.softmax(pred_logits, dim=1)

#         # Compute cross-entropy loss with smoothed labels
#         loss = (-smoothed_probs * torch.log(pred_probs + 1e-9)).sum(dim=1).mean()

#         return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import quadtree_centroids

def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371):
    lat1, lon1, lat2, lon2 = map(lambda x: x * (torch.pi / 180), [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))
    return earth_radius * c

class HaversineSmoothedLoss(nn.Module):
    def __init__(self, tau=75.0):
        super(HaversineSmoothedLoss, self).__init__()
        self.tau = tau

        # Ensure centroids are stored as a tensor on the correct device
        global quadtree_centroids_tensor
        quadtree_centroids_tensor = torch.tensor(quadtree_centroids, dtype=torch.float32)
        
    def forward(self, pred_logits, latlon, geocell_indices):
        batch_size, num_classes = pred_logits.shape
        device = pred_logits.device

        # Move centroids to device
        global quadtree_centroids_tensor
        quadtree_centroids_tensor = quadtree_centroids_tensor.to(device)

        # Ensure latlon is on the correct device
        latlon = torch.tensor(latlon, dtype=torch.float32).to(device) if isinstance(latlon, list) else latlon.to(device)

        # Get true geocell coordinates
        true_geocell_coords = quadtree_centroids_tensor[geocell_indices]  # (batch_size, 2)

        # Compute distances
        lat1, lon1 = latlon[:, 0].unsqueeze(1), latlon[:, 1].unsqueeze(1)  # (batch_size, 1)
        lat2, lon2 = quadtree_centroids_tensor[:, 0].unsqueeze(0), quadtree_centroids_tensor[:, 1].unsqueeze(0)  # (1, num_classes)

        dists = haversine_distance(lat1, lon1, lat2, lon2)  # (batch_size, num_classes)
        d_true = haversine_distance(latlon[:, 0], latlon[:, 1], true_geocell_coords[:, 0], true_geocell_coords[:, 1])  # (batch_size,)

        # Compute smoothed probabilities
        smoothed_probs = torch.exp(-(dists - d_true.unsqueeze(1)) / self.tau)  
        smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=1, keepdim=True) 

        # Use log-softmax for numerical stability
        loss = (-smoothed_probs * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()

        return loss
