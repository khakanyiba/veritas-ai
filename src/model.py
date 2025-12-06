import torch
import torch.nn as nn
import torch.nn.functional as F

class VeritasCNN(nn.Module):
    def __init__(self):
        super(VeritasCNN, self).__init__()
        
        # 1. Convolutional Block 1 (The "Feature Extractor")
        # Input: 32x32x3 (RGB)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2) # Downsample to 16x16
        
        # 2. Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # MaxPool will downsample to 8x8
        
        # 3. Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # MaxPool will downsample to 4x4
        
        # 4. Fully Connected Layers (The "Decision Maker")
        # Input features: 128 channels * 4 * 4 size
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5) # Prevents overfitting
        self.fc2 = nn.Linear(512, 1)   # Output: 1 score (Real vs Fake)

    def forward(self, x):
        # Pass through Conv blocks with ReLU activation
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 32 -> 16
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 16 -> 8
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 8 -> 4
        
        # Flatten for the decision layer
        x = x.view(-1, 128 * 4 * 4)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x) # Squash output between 0 and 1