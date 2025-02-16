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
import json


# data

BATCH_SIZE = 64
NUM_WORKERS = 4

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# dataset = OSV5MDataset(transform=transform)
train_dataset = OSV5MDataset(split="train", transform=transform)
val_dataset = OSV5MDataset(split="val", transform=transform)

print("after dataset")

# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

NUM_CLASSES = len(quadtree_centroids)
print(f"Number of unique geocells: {NUM_CLASSES}")

# Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GeocellResNet(NUM_CLASSES).to(device)
criterion = HaversineSmoothedLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    # Labels contain geocell probabilities
    for images, geocells_smoothed, latlon_of_image, geocell_index in dataloader:
        images = images.to(device)
        geocells_smoothed = geocells_smoothed.to(device)  

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, latlon_of_image, geocell_index)  # Use indices instead of raw geocells
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, geocells_smoothed, latlon_of_image, geocell_index in val_loader:
            images, geocells_smoothed = images.to(device), geocells_smoothed.to(device)
            outputs = model(images)
            loss = criterion(outputs, latlon_of_image, geocell_index)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # epoch_loss = total_loss / len(dataloader)
    # print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss}")

    # Save model after each epoch
    model_path = f"geocell_resnet50_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Checkpoint saved: {model_path}")

# Load test dataset
test_dataset = OSV5MDataset(split="test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Evaluate on test set
print("Evaluation on test set...")
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

# Final model save
torch.save(model.state_dict(), "geocell_resnet50_final.pth")
print("Training complete! Final model saved.")
