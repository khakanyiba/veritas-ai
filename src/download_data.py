import os
import zipfile
from pathlib import Path

# Config
DATA_DIR = Path("data")
# We look for any zip file in the data folder
ZIP_FILES = list(DATA_DIR.glob("*.zip"))

def setup_data():
    """Extracts the manually downloaded CIFAKE dataset."""
    
    if not ZIP_FILES:
        print(f"No zip file found in {DATA_DIR}")
        print("Please download the dataset from Kaggle and place it in the 'data' folder.")
        return

    zip_path = ZIP_FILES[0] # Take the first zip found
    print(f"Found dataset: {zip_path.name}")
    print("Extracting files... (This might take a moment)")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        print(f"Extraction Complete!")
        
        # Verify structure
        train_dir = DATA_DIR / "train"
        test_dir = DATA_DIR / "test"
        
        if train_dir.exists() and test_dir.exists():
            print(f"Verified: Found 'train' and 'test' folders.")
            # Optional: Delete zip to save space
            # os.remove(zip_path) 
        else:
            print("Warning: Folder structure looks unexpected. Check the 'data' folder.")
            
    except zipfile.BadZipFile:
        print("Error: The zip file appears to be corrupted.")

if __name__ == "__main__":
    setup_data()