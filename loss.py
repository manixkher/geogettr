import torch
import torch.nn as nn
import numpy as np
# NOTES TO DO --
# USE GEOCELL CENTROIDS FROM QUADTREE_1000 TO CALCULATE HAVERSINE LOSS. RIGHT NOW IT DOESN'T MAKE SENSE
def haversine_distance(coord1, coord2, earth_radius=6371):
    """Computes the Haversine distance between two lat/lon points."""
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius * c  # Returns distance in km

class HaversineSmoothedLoss(nn.Module):
    """Haversine Smoothed Classification Loss from PIGEON."""
    def __init__(self, geocell_mapping, tau=75.0):
        super(HaversineSmoothedLoss, self).__init__()
        self.geocell_mapping = geocell_mapping
        self.tau = tau  # Temperature parameter

   
    def forward(self, pred_logits, latlon, geocell_indices):
        """Computes Haversine-smoothed classification loss."""
        batch_size, num_classes = pred_logits.shape
        pred_probs = torch.softmax(pred_logits, dim=1)  # Convert logits to probabilities

        # Ensure latlon is a Tensor and convert it to a list of tuples (lat, lon)
        if isinstance(latlon, torch.Tensor):
            latlon = latlon.cpu().numpy()  # Convert tensor to NumPy array

        elif isinstance(latlon, list) and isinstance(latlon[0], torch.Tensor):
            # Convert a list of tensors (lat, lon) into a proper list of tuples
            latlon = [(lat.item(), lon.item()) for lat, lon in zip(latlon[0], latlon[1])]

        print("Converted latlon:", latlon)  # Debugging print

        if not isinstance(latlon, list):
            raise TypeError(f"Expected list of geocell coordinates, but got {type(latlon)}")
        
        true_coords_list = []
        for coord in latlon:
            if isinstance(coord, tuple) and len(coord) == 2:
                true_coords_list.append(np.array(coord))  # Convert (lat, lon) to NumPy array
            else:
                raise TypeError(f"Expected (x, y) coordinate tuple, but got {coord} (type: {type(coord)})")

        # Compute distance-aware soft targets
        target_probs = torch.zeros_like(pred_probs, device=pred_logits.device)

        for i, true_coord in enumerate(true_coords_list):
            for j, geo_coord in enumerate(geocell_indices):
                geo_coord = np.array(geo_coord)
                dist = haversine_distance(true_coord, geo_coord)
                target_probs[i, j] = np.exp(-dist / self.tau)  # Apply smoothing

        # Normalize target probabilities
        target_probs /= target_probs.sum(dim=1, keepdim=True)

        # Compute cross-entropy loss with smoothed labels
        loss = (-target_probs * torch.log(pred_probs + 1e-9)).sum(dim=1).mean()

        return loss
