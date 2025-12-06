import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
import random

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"

def show_comparison():
    # 1. Get random images
    real_path = TRAIN_DIR / "REAL"
    fake_path = TRAIN_DIR / "FAKE"
    
    real_img_name = random.choice(os.listdir(real_path))
    fake_img_name = random.choice(os.listdir(fake_path))
    
    real_img = Image.open(real_path / real_img_name)
    fake_img = Image.open(fake_path / fake_img_name)
    
    # 2. Plot them
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(real_img)
    axes[0].set_title(f"REAL\n{real_img_name}")
    axes[0].axis("off")
    
    axes[1].imshow(fake_img)
    axes[1].set_title(f"AI GENERATED\n{fake_img_name}")
    axes[1].axis("off")
    
    plt.show()
    print("Plot generated. Check the popup window.")

if __name__ == "__main__":
    show_comparison()