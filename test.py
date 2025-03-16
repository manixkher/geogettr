import argparse
import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import OSV5MDataset, quadtree_centroids
from model import GeocellResNet
from loss import HaversineSmoothedLoss, haversine_distance

def parse_args():
    parser = argparse.ArgumentParser(description="Test geogettr model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--label_map", type=str, required=True, help="Path to the label mapping pickle file")
    return parser.parse_args()

def compute_haversine(pred_index, true_index):
    # Retrieve the (lat, lon) pair for both predicted and true indices.
    pred_lat, pred_lon = quadtree_centroids[pred_index].cpu().tolist()
    true_lat, true_lon = quadtree_centroids[true_index].cpu().tolist()
    # Compute Haversine distance (extracting the scalar value with .item())
    return haversine_distance(
        torch.tensor(pred_lat), torch.tensor(pred_lon),
        torch.tensor(true_lat), torch.tensor(true_lon)
    ).item()

def main():
    args = parse_args()
    
    # Set up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load label mapping.
    with open(args.label_map, "rb") as f:
        label_to_index = pickle.load(f)
    print("Loaded label mapping:", label_to_index)
    
    # Define image transforms.
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    # ])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load test dataset.
    test_dataset = OSV5MDataset(split="test", transform=transform, dataset_path=args.dataset, label_to_index=label_to_index)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Load the model.
    NUM_CLASSES = quadtree_centroids.shape[0] if isinstance(quadtree_centroids, torch.Tensor) else len(quadtree_centroids)
    model = GeocellResNet(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully. Running inference...")
    
    # Initialize loss criterion.
    criterion = HaversineSmoothedLoss()
    
    # Variables to accumulate Haversine distance and loss.
    total_distance = 0.0
    total_loss = 0.0
    sample_count = 0  # Total number of samples processed
    
    # Inference loop: compute predictions, distances, and loss in one pass.
    with torch.no_grad():
        for batch_idx, (images, geocells_smoothed, latlon_of_image, geocell_index) in enumerate(test_loader):
            images = images.to(device)
            geocell_index = geocell_index.to(device)  # True geocell indices
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Predicted geocell indices.
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            true_indices = geocell_index.cpu().numpy()
            
            # Compute Haversine distances for the current batch.
            batch_distances = [compute_haversine(int(pred), int(true)) for pred, true in zip(predictions, true_indices)]
            total_distance += sum(batch_distances)
            sample_count += len(batch_distances)
            
            # Compute loss for the current batch.
            loss = criterion(outputs, latlon_of_image, geocell_index, geocells_smoothed)
            total_loss += loss.item()
            
            # Optionally print per-sample details.
            for j in range(len(predictions)):
                print(f"Sample {j+1 + batch_idx*test_loader.batch_size}: True Geocell = {true_indices[j]}, "
                      f"True Latlon = {latlon_of_image[j]}, Predicted Geocell = {predictions[j]}, "
                      f"Haversine Distance = {batch_distances[j]:.2f} km")
    
    avg_distance = total_distance / sample_count if sample_count > 0 else float("inf")
    avg_loss = total_loss / len(test_loader)  # Average loss per batch
    
    print(f"\nFinal Average Haversine Distance (over {sample_count} samples): {avg_distance:.2f} km")
    print(f"Final Test Loss: {avg_loss:.4f}")
    print("Inference complete.")

if __name__ == '__main__':
    main()
