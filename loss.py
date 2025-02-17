import torch
import torch.nn as nn
import numpy as np
from data import quadtree_centroids
# NOTES TO DO --
# USE GEOCELL CENTROIDS FROM QUADTREE_1000 TO CALCULATE HAVERSINE LOSS. RIGHT NOW IT DOESN'T MAKE SENSE
def haversine_distance(coord1, coord2, earth_radius=6371):
    """Computes the Haversine distance between two lat/lon points."""
    
    # lat1, lon1 = np.radians(coord1)
    # lat2, lon2 = np.radians(coord2)

    # dlat = lat2 - lat1
    # dlon = lon2 - lon1

    # a = (np.sin(dlat / 2) ** 2) + (np.cos(lat1) * np.cos(lat2)) * (np.sin(dlon / 2) ** 2)
    # c = 2 * np.arcsin(np.sqrt(a))

    """Computes the Haversine distance between two lat/lon points using PyTorch (GPU-accelerated)."""

    lat1, lon1, lat2, lon2 = map(torch.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    return earth_radius * c  # Returns distance in km

class HaversineSmoothedLoss(nn.Module):
    def __init__(self, tau=75.0):
        super(HaversineSmoothedLoss, self).__init__()
        self.tau = tau

        # Convert quadtree centroids to a single GPU tensor for efficiency
        global quadtree_centroids_tensor, quadtree_ids_tensor
        quadtree_centroids_tensor = quadtree_centroids
        quadtree_ids_tensor = quadtree_cluster_ids
    

    # def forward(self, pred_logits, latlon, geocell_indices):
    #     """
    #     Compute the Haversine-smoothed classification loss.

    #     Args:
    #         pred_logits (Tensor): Predicted logits of shape (batch_size, num_classes).
    #         latlon (list of tuples): List of (latitude, longitude) coordinates for each sample in the batch.
    #         geocell_indices (Tensor): Tensor of shape (batch_size,) containing the true geocell IDs.

    #     Returns:
    #         Tensor: Loss value.
    #     """
    #     batch_size, num_classes = pred_logits.shape
    #     pred_probs = torch.softmax(pred_logits, dim=1)

    #     if isinstance(latlon, torch.Tensor):
    #         latlon = latlon.cpu().numpy()
    #     elif isinstance(latlon, list) and isinstance(latlon[0], torch.Tensor):
    #         latlon = [(lat.item(), lon.item()) for lat, lon in zip(latlon[0], latlon[1])]

    #     true_coords_list = [np.array(coord) for coord in latlon]
        
    #     # Initialize soft target probabilities
    #     target_probs = torch.zeros_like(pred_probs, device=pred_logits.device)

    #     # Compute soft labels using distances from true coords to ALL geocell centroids
    #     for i, true_coord in enumerate(true_coords_list):
    #         smoothed_values = {}
    #         counter = 0
            
    #         true_geocell_id = geocell_indices[i].item()
    #         true_geocell_coord = np.array(quadtree_centroids[true_geocell_id])
    #         dist_n = haversine_distance(true_coord, true_geocell_coord)
            
    #         for j, (geo_id, geo_coord) in enumerate(quadtree_centroids.items()):
    #             counter += 1
                
    #             geo_coord = np.array(geo_coord)
                
    #             dist_i = haversine_distance(true_coord, geo_coord)
                
                
    #             smoothed_values[j] = np.exp(-(dist_i - dist_n) / self.tau)  # Apply exponential smoothing

    #         # Convert smoothed distances into probability distribution
    #         total = sum(smoothed_values.values())
            
            
    #         for j in smoothed_values:
    #             target_probs[i, j] = smoothed_values[j] / total  # Normalize
       
    #     # Compute cross-entropy loss with smoothed labels
    #     loss = (-target_probs * torch.log(pred_probs + 1e-9)).sum(dim=1).mean()

    #     return loss

    # Forward function without using NumPy operations or for loops
    def forward(self, pred_logits, latlon, geocell_indices):
        """
        Compute the Haversine-smoothed classification loss.

        Args:
            pred_logits (Tensor): Predicted logits of shape (batch_size, num_classes).
            latlon (Tensor): Tensor of shape (batch_size, 2) with (latitude, longitude).
            geocell_indices (Tensor): Tensor of shape (batch_size,) containing the true geocell IDs.

        Returns:
            Tensor: Loss value.
        """
        batch_size, num_classes = pred_logits.shape
        device = pred_logits.device  # Ensure computations stay on the correct device

        # Ensure latlon is a tensor on the correct device
        latlon = torch.tensor(latlon, dtype=torch.float32, device=device) if isinstance(latlon, list) else latlon

        # Get true geocell coordinates for each batch sample
        true_geocell_coords = quadtree_centroids_tensor[geocell_indices]  # Shape: (batch_size, 2)

        # Compute Haversine distances between each sample and ALL geocells (vectorized)
        lat1, lon1 = latlon[:, 0].unsqueeze(1), latlon[:, 1].unsqueeze(1)  # Shape: (batch_size, 1)
        lat2, lon2 = quadtree_centroids_tensor[:, 0], quadtree_centroids_tensor[:, 1]  # Shape: (num_classes,)

        dists = haversine_distance(lat1, lon1, lat2, lon2, self.earth_radius)  # Shape: (batch_size, num_classes)

        # Get distance to the true geocell (vectorized)
        d_true = haversine_distance(latlon[:, 0], latlon[:, 1], true_geocell_coords[:, 0], true_geocell_coords[:, 1], self.earth_radius)  # Shape: (batch_size,)

        # Compute smoothed probabilities
        smoothed_probs = torch.exp(-(dists - d_true.unsqueeze(1)) / self.tau)  # Shape: (batch_size, num_classes)
        # Normalisation
        smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=1, keepdim=True) 

        # Compute softmax predictions
        pred_probs = torch.softmax(pred_logits, dim=1)

        # Compute cross-entropy loss with smoothed labels
        loss = (-smoothed_probs * torch.log(pred_probs + 1e-9)).sum(dim=1).mean()

        return loss
