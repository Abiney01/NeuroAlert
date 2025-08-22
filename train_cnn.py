import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # Import tqdm for the progress bar
import os

# --- PyTorch GPU/CPU Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Configuration ---
DATASET_PATH = r'D:\Internship\NeuroAlert\Driver Drowsiness Dataset (DDD)'
IMAGE_SIZE = (227, 227)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_NAME = 'drowsiness_model.pth'

# --- Model Architecture ---
class DrowsinessCNN(nn.Module):
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_pytorch_model():
    # --- Check for GPU and exit if not found ---
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA-enabled GPU detected. Training will not proceed on CPU.")
        print("[INFO] Please check your PyTorch installation and CUDA drivers.")
        return

    # --- Data Loading and Transformation ---
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # Added num_workers for better CPU utilization
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, and Optimizer ---
    model = DrowsinessCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Use tqdm to show a progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for i, data in enumerate(train_loader_tqdm, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Update the progress bar with the current loss
            train_loader_tqdm.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader):.4f}")

    print("Training finished. Saving model...")
    torch.save(model.state_dict(), MODEL_NAME)
    print(f"Model saved to {MODEL_NAME}")

if __name__ == '__main__':
    train_pytorch_model()