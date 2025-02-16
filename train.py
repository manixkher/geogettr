import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import OSV5MDataset  # Ensure the dataset class is correctly named
from data import quadtree_centroids
from loss import HaversineSmoothedLoss
from model import GeocellResNet
from torch.utils.data import random_split
import json
from tqdm import tqdm
import argparse
print("Beginning train.py")
parser = argparse.ArgumentParser(description="Geogettr training params")
parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs")

# data

BATCH_SIZE = 64
NUM_WORKERS = 2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# dataset = OSV5MDataset(transform=transform)
dataset = OSV5MDataset(split="train", transform=transform, dataset_path = args.dataset)
train_size = int(0.9 * len(dataset))  # 80% training
val_size = len(dataset) - train_size  # 20% validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("after dataset")

# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

NUM_CLASSES = len(quadtree_centroids)
print(f"Number of unique geocells: {NUM_CLASSES}")

# Training

model = GeocellResNet(NUM_CLASSES).to(device)

if num_gpus > 1:
    print(f"Using DataParallel across {num_gpus} GPUs")
    model = nn.DataParallel(model)

criterion = HaversineSmoothedLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

EPOCHS = args.epochs

for epoch in range(EPOCHS):

    model.train()
    total_train_loss = 0.0

    # Labels contain geocell probabilities
    batch_counter = 0
    train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    for images, geocells_smoothed, latlon_of_image, geocell_index in train_loader:
        images = images.to(device)
        geocells_smoothed = geocells_smoothed.to(device)

        latlon_of_image = latlon_of_image.to(device)
        geocell_index = geocell_index.to(device) 

        optimizer.zero_grad()
        outputs = model(images)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, latlon_of_image, geocell_index)  # Use indices instead of raw geocells
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_progress.set_postfix(loss=f"{loss.item():.4f}")
        
        if (batch_counter + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Batch {batch_counter+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
    
    model.eval()
    total_val_loss = 0.0
    val_progress = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
    with torch.no_grad():
        for images, geocells_smoothed, latlon_of_image, geocell_index in val_loader:
            images, geocells_smoothed = images.to(device), geocells_smoothed.to(device)
            latlon_of_image = latlon_of_image.to(device)
            geocell_index = geocell_index.to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0] 
            
            loss = criterion(outputs, latlon_of_image, geocell_index)
            total_val_loss += loss.item()
            val_progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save model after each epoch
    model_path = f"geocell_resnet50_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Checkpoint saved: {model_path}")

# Final model save
torch.save(model.state_dict(), "geocell_resnet50_final.pth")
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


