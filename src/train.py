import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import VeritasCNN
import time

# Config
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5 # Number of times to review the whole dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print(f"Training on: {DEVICE}")
    
    # 1. Prepare Data (Augmentation + Normalization)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values to -1..1
    ])

    print("Loading Data...")
    train_data = datasets.ImageFolder(root='data/train', transform=transform)
    test_data = datasets.ImageFolder(root='data/test', transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    model = VeritasCNN().to(DEVICE)
    criterion = nn.BCELoss() # Binary Cross Entropy (for 2 classes)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)
            labels = labels.unsqueeze(1) # Fix shape for BCELoss

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        print(f"Epoch {epoch + 1} Complete")

    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # 4. Save the Brain
    torch.save(model.state_dict(), "veritas_model.pth")
    print("Model saved as 'veritas_model.pth'")

if __name__ == "__main__":
    train()