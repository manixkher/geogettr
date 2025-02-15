import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import OSV5MDataset  # Ensure the dataset class is correctly named
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

dataset = OSV5MDataset(transform=transform)
print("after dataset")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
print("after dataloader, starting mapping")

# ============================
# Geocell Label Mapping
# ============================
geocell_mapping = {}
geocell_labels = []

for sample in dataset:
    image, smoothed_geocell, latlon, geocell = sample  # label is a tensor of probabilities
    print("geocell")
    print(geocell)
    # Convert label (soft probability distribution) to its most likely geocell
    geocell_index = torch.argmax(smoothed_geocell).item()  # Get the index of the highest probability geocell
    
    if geocell_index not in geocell_mapping:
        geocell_mapping[geocell_index] = latlon  # Assign a unique ID
    
    geocell_labels.append(geocell_mapping[geocell_index])  # Save the mapped label

print("Saving mapping...")
print(geocell_mapping)

# Save mapping
with open("geocell_mapping.json", "w") as f:
    json.dump(geocell_mapping, f)
    

NUM_CLASSES = len(geocell_mapping)
print(f"Number of unique geocells: {NUM_CLASSES}")


# Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GeocellResNet(NUM_CLASSES).to(device)
criterion = HaversineSmoothedLoss(geocell_mapping)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images, geocells_smoothed, latlon_of_image, geocell = batch  # Labels contain geocell probabilities
        images = images.to(device)
        geocells_smoothed = geocells_smoothed.to(device)  

        optimizer.zero_grad()
        outputs = model(images)

        # 🔹 Convert smoothed geocell labels to discrete indices
        geocell_indices = torch.argmax(geocells_smoothed, dim=1)  # Get most probable geocell

        loss = criterion(outputs, latlon_of_image, geocell_indices)  # Use indices instead of raw geocells
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss}")



    # Save model after each epoch
    model_path = f"geocell_resnet50_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Checkpoint saved: {model_path}")

# Final model save
torch.save(model.state_dict(), "geocell_resnet50_final.pth")
print("Training complete! Final model saved.")
