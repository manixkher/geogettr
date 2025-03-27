import torch
import torch.nn as nn
import numpy as np
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

        centroids_tensor = torch.tensor(quadtree_centroids, dtype=torch.float32)
        # Centroids to be accessed by other portions of code.
        self.register_buffer('centroids', centroids_tensor)
        

    def forward(self, pred_logits, latlon, geocell_indices, smoothed_labels):

            loss = (-smoothed_labels * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()
            print("Computed loss:", loss.item())
            print("==================")

            return loss
