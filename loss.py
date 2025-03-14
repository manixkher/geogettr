import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from data import quadtree_centroids



# NOTES TO DO --
# USE GEOCELL CENTROIDS FROM QUADTREE_1000 TO CALCULATE HAVERSINE LOSS. RIGHT NOW IT DOESN'T MAKE SENSE
def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371):


    """Computes the Haversine distance between two lat/lon points using PyTorch (GPU-accelerated)."""

    # lat1, lon1, lat2, lon2 = map(torch.radians, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = map(lambda x: x * (torch.pi / 180), [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    return earth_radius * c  # Returns distance in km

def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371):
    lat1, lon1, lat2, lon2 = map(lambda x: x * (torch.pi / 180), [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))
    return earth_radius * c

class HaversineSmoothedLoss(nn.Module):
    def __init__(self, tau=85.0):
        super(HaversineSmoothedLoss, self).__init__()
        self.tau = tau

        # Ensure centroids are stored as a tensor on the correct device
        centroids_tensor = torch.tensor(quadtree_centroids, dtype=torch.float32)
        self.register_buffer('centroids', centroids_tensor)
        
    # def forward(self, pred_logits, latlon, geocell_indices, smoothed_labels):
    #     batch_size, num_classes = pred_logits.shape
    #     device = pred_logits.device

    #     # Move centroids to device
    #     quadtree_centroids_tensor = self.centroids

    #     # Ensure latlon is on the correct device
    #     latlon = torch.tensor(latlon, dtype=torch.float32).to(device) if isinstance(latlon, list) else latlon.to(device)

    #     # Get true geocell coordinates
    #     true_geocell_coords = quadtree_centroids_tensor[geocell_indices]  # (batch_size, 2)

    #     # Compute distances
    #     lat1, lon1 = latlon[:, 0].unsqueeze(1), latlon[:, 1].unsqueeze(1)  # (batch_size, 1)
    #     lat2, lon2 = quadtree_centroids_tensor[:, 0].unsqueeze(0), quadtree_centroids_tensor[:, 1].unsqueeze(0)  # (1, num_classes)

    #     dists = haversine_distance(lat1, lon1, lat2, lon2)  # (batch_size, num_classes)
    #     d_true = haversine_distance(latlon[:, 0], latlon[:, 1], true_geocell_coords[:, 0], true_geocell_coords[:, 1])  # (batch_size,)
        
    #     # print("=== Debug Info ===")
    #     # print("Input latlon:", latlon.cpu().numpy())
    #     # print("True geocell coordinates:", true_geocell_coords.cpu().numpy())
    #     # print("True geocell label:", geocell_indices.cpu().numpy())  # Added line
    #     # print("Distance matrix (dists) shape:", dists.shape)
    #     # print("dists (first sample):", dists[0].cpu().numpy())
    #     # print("d_true:", d_true.cpu().numpy())

    #     # Compute smoothed probabilities
    #     smoothed_probs = torch.exp(-(dists - d_true.unsqueeze(1)) / self.tau)  
    #     smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=1, keepdim=True) 
    #     # print("Smoothed probabilities (first sample):", smoothed_probs[0].cpu().numpy())
    #     max_val, max_idx = smoothed_probs[0].max(dim=0)
    #     # Get the top 5 smoothed probabilities and their corresponding indices for each sample
    #     top_values, top_indices = smoothed_probs.topk(1, dim=1)
    #     # print("Top 5 smoothed probabilities for each sample:")
    #     # print(top_values.cpu().numpy())
    #     # print("Corresponding indices:")
    #     # print(top_indices.cpu().numpy())


    #     # Use log-softmax for numerical stability
    #     loss = (-smoothed_probs * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()
    #     print("Computed loss:", loss.item())
    #     print("==================")

    #     return loss

    def forward(self, pred_logits, latlon, geocell_indices, smoothed_labels):

            # Use log-softmax for numerical stability
            loss = (-smoothed_labels * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()
            print("Computed loss:", loss.item())
            print("==================")

            return loss
