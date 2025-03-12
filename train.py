import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import OSV5MDataset  # Ensure the dataset class is correctly named
from data import quadtree_centroids
from loss import HaversineSmoothedLoss
from model import GeocellResNet
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
import json
from tqdm import tqdm
import argparse
if __name__ == "__main__":
    print("Beginning train.py")
    # mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Geogettr training params")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--model_name", type=str, required=True, help="Final model name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    # data

    BATCH_SIZE = 128
    NUM_WORKERS = 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # dataset = OSV5MDataset(transform=transform)
    df = pd.read_csv("/home/s2751435/Work/geogettr/my_datasets/osv5m/reduced_train_europe.csv")
    all_labels_full = df["quadtree_10_1000"].tolist()
    raw_labels = df["quadtree_10_1000"].tolist()
    unique_labels, counts = np.unique(raw_labels, return_counts=True)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    print("Total unique labels in CSV:", len(unique_labels))
    print("First 10 mappings:")
    for label in unique_labels[:10]:
        print(f"Original label: {label} -> Mapped: {label_to_index[label]}")
    with open(f"{args.model_name}_label_to_index.pkl", "wb") as f:
        pickle.dump(label_to_index, f)

    print("Mapping saved to label_to_index.pkl")
    dataset = OSV5MDataset(split="train", transform=transform, dataset_path = args.dataset, label_to_index=label_to_index)
    train_size = int(0.9 * len(dataset))  # 80% training
    val_size = len(dataset) - train_size  # 20% validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # df = pd.read_csv("/home/s2751435/Work/geogettr/my_datasets/osv5m/reduced_train_europe.csv")
    # all_labels_full = df["quadtree_10_1000"].tolist()
    print("after dataset")
    
    # Debug: print lengths and sampler indices before DataLoader creation
    print("Full dataset length:", len(dataset))  # OSV5MDataset length
    print("Training subset length:", len(train_dataset))
    print("Validation subset length:", len(val_dataset))

    # Assuming df from CSV is already loaded and all_labels_full is computed:
    print("CSV length:", len(all_labels_full))

    # Print the training indices from the Subset
    train_indices = list(train_dataset.indices)
    print("Training indices from random_split (first 20):", train_indices[:20])
    print("Training indices from random_split, min:", min(train_indices), "max:", max(train_indices))

    # We computed raw_labels using these indices from the CSV:
    raw_labels = [all_labels_full[i] for i in train_indices]
    print("Raw labels length:", len(raw_labels))
    
    
    # df = pd.read_csv("/home/s2751435/Work/geogettr/my_datasets/osv5m/reduced_train_europe.csv")
    # Assume the geocell label is stored in a column named 'quadtree_10_1000'
    # raw_labels = df["quadtree_10_1000"].tolist()

    # Compute frequency counts for each label
    # unique_labels, counts = np.unique(raw_labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, counts))

    print("Unique classes seen in CSV:", len(unique_labels))
    
    print("Label mapping:", label_to_index)
    # Compute the weight for each sample (inverse frequency)
    weights = [1.0 / label_to_count[label] for label in raw_labels]
    weights = torch.DoubleTensor(weights)
    
    

    
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    print("Weights vector length:", len(weights))

    # Now, sample a few indices from the sampler:
    sampled_indices = list(iter(sampler))
    print("Sampler produced", len(sampled_indices), "indices.")
    if sampled_indices:
        print("Sampler indices: min =", min(sampled_indices), "max =", max(sampled_indices))
        # These indices should be in the range [0, len(train_dataset)-1]
        if max(sampled_indices) >= len(train_dataset):
            print("Error: sampler index exceeds training subset length!")
    
    print("after sampler")

    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    NUM_CLASSES = len(quadtree_centroids)
    print(f"Number of unique geocells: {NUM_CLASSES}")

    # Training

    model = GeocellResNet(NUM_CLASSES).to(device)

    # if num_gpus > 1:
    #     print(f"Using DataParallel across {num_gpus} GPUs")
    #     model = nn.DataParallel(model)

    criterion = HaversineSmoothedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    # Set up learning rate annealing using StepLR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    EPOCHS = args.epochs

    for epoch in range(EPOCHS):

        model.train()
        total_train_loss = 0.0

        # Training Progress Bar (Fixed)
        train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        batch_idx = 0
        for images, geocells_smoothed, latlon_of_image, geocell_index in train_progress:
            
            images, geocells_smoothed = images.to(device), geocells_smoothed.to(device)
            latlon_of_image = latlon_of_image.to(device)
            geocell_index = geocell_index.to(device) 

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Handle DataParallel output

            loss = criterion(outputs, latlon_of_image, geocell_index)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_progress.set_postfix(loss=f"{loss.item():.4f}")  # Update tqdm

            # Inside your training loop, after computing outputs:
            if batch_idx % 500 == 0:  # adjust frequency as needed
                # If outputs is a tuple, extract the logits
                current_logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                
                # Compute softmax probabilities
                softmax_probs = F.softmax(current_logits, dim=1)
                
                # Print summary statistics for logits
                logits_min = current_logits.min().item()
                logits_max = current_logits.max().item()
                logits_mean = current_logits.mean().item()
                logits_std = current_logits.std().item()
                
                # Print summary statistics for softmax probabilities
                probs_min = softmax_probs.min().item()
                probs_max = softmax_probs.max().item()
                probs_mean = softmax_probs.mean().item()
                probs_std = softmax_probs.std().item()
                
                # Optionally, compute the average per-class standard deviation across the batch
                avg_per_class_std = softmax_probs.std(dim=0).mean().item()
                
                print(f"Batch {batch_idx} debug:")
                print(f"  Logits -> min: {logits_min:.4f}, max: {logits_max:.4f}, mean: {logits_mean:.4f}, std: {logits_std:.4f}")
                print(f"  Softmax -> min: {probs_min:.4f}, max: {probs_max:.4f}, mean: {probs_mean:.4f}, std: {probs_std:.4f}")
                print(f"  Avg per-class softmax std across batch: {avg_per_class_std:.4f}")
            batch_idx += 1


        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

        # -----------------------------
        # VALIDATION PHASE (Fixed tqdm)
        # -----------------------------
        model.eval()
        total_val_loss = 0.0
        val_progress = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for images, geocells_smoothed, latlon_of_image, geocell_index in val_progress:
                images, geocells_smoothed = images.to(device), geocells_smoothed.to(device)
                latlon_of_image = latlon_of_image.to(device)
                geocell_index = geocell_index.to(device)

                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Handle DataParallel output

                loss = criterion(outputs, latlon_of_image, geocell_index)
                total_val_loss += loss.item()
                val_progress.set_postfix(loss=f"{loss.item():.4f}")  # Update tqdm

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation | Val Loss: {avg_val_loss:.4f}")
        scheduler.step()
        # Save model after each epoch
        model_path = f"epochs/{args.model_name}_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Checkpoint saved: {model_path}")

    # Final model save
    torch.save(model.state_dict(), f"final_models/{args.model_name}_final.pth")
    print("Training complete! Final model saved.")

    # Load test dataset
    test_dataset = OSV5MDataset(split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Evaluate on test set
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for images, geocells_smoothed, latlon_of_image, geocell_index in test_loader:
            images, geocells_smoothed = images.to(device), geocells_smoothed.to(device)
            outputs = model(images)
            loss = criterion(outputs, latlon_of_image, geocell_index)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")


