import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from model import DigitCNN

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 0.001
NPZ_PATH = "ocr_dataset.npz"
MODEL_PATH = "ocr_model.pth"
NUM_CLASSES = 85

# Optimized Custom PyTorch Dataset
# Isme hum images ko __init__ ke waqt hi PyTorch Tensors me convert aur normalize kar lete hain.
# Isse dataloader loop ke andar PIL conversions aur dynamic transforms ki heavy processing completely bypass ho jati hai.
# Yeh training speed ko CPU par 10x-20x fast kar dega.
class OCRDataset(Dataset):
    def __init__(self, npz_path, split="train", train_ratio=0.8):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"'{npz_path}' nahi mila! Pehle 'python dataset_generator.py' run karein.")
            
        data = np.load(npz_path)
        images = data["images"]
        labels = data["labels"]
        
        # Train-test split
        num_samples = len(images)
        indices = np.arange(num_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * train_ratio)
        if split == "train":
            selected_images = images[indices[:split_idx]]
            selected_labels = labels[indices[:split_idx]]
        else:
            selected_images = images[indices[split_idx:]]
            selected_labels = labels[indices[split_idx:]]
            
        # [CRITICAL OPTIMIZATION]
        # Images ko float32 me change karenge aur [0, 1] me scale karenge
        float_images = selected_images.astype(np.float32) / 255.0
        
        # Standard Normalization: Mean (0.1307) aur Std (0.3081)
        normalized_images = (float_images - 0.1307) / 0.3081
        
        # PyTorch format ke liye Channel dimension [1, 28, 28] expand karenge
        normalized_images = np.expand_dims(normalized_images, axis=1)
        
        # NumPy to PyTorch tensors in one go (memory sharing)
        self.tensor_images = torch.from_numpy(normalized_images)
        self.tensor_labels = torch.from_numpy(selected_labels)
        
    def __len__(self):
        return len(self.tensor_images)
        
    def __getitem__(self, idx):
        # Kuch bhi heavy function yahan execute nahi hoga, direct memory slice return hogi!
        return self.tensor_images[idx], self.tensor_labels[idx]

def check_accuracy(loader, model, criterion, device):
    """
    Test/Validation dataset par accuracy check karne ka helper.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False, unit="batch")
        for data, targets in pbar:
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * data.size(0)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    mean_loss = total_loss / total
    accuracy = 100. * correct / total
    return mean_loss, accuracy

def train():
    parser = argparse.ArgumentParser(description="OCR Model Training Script")
    parser.add_argument("--force", action="store_true", help="Force retraining even if model weights exist")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check karenge agar model pehle se hi trained hai aur save ho chuka hai
    if os.path.exists(MODEL_PATH) and not args.force:
        print(f"\n[INFO] '{MODEL_PATH}' pehle se exist karta hai! Training skip ki ja rahi hai.")
        print("Existing model ko load karke evaluate karte hain...")
        
        model = DigitCNN(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        try:
            test_dataset = OCRDataset(npz_path=NPZ_PATH, split="test")
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            criterion = nn.CrossEntropyLoss()
            
            test_loss, test_acc = check_accuracy(test_loader, model, criterion, device)
            print(f"\n==========================================")
            print(f"LOADED OCR MODEL ACCURACY : {test_acc:.2f}%")
            print(f"LOADED OCR MODEL LOSS     : {test_loss:.4f}")
            print(f"==========================================\n")
        except FileNotFoundError as e:
            print(e)
            print("Pehle dataset generate karein 'python dataset_generator.py'")
        return

    print("\nOCR Dataset ko load aur optimize kiya ja raha hai...")
    try:
        train_dataset = OCRDataset(npz_path=NPZ_PATH, split="train")
        test_dataset = OCRDataset(npz_path=NPZ_PATH, split="test")
    except FileNotFoundError as e:
        print(e)
        print("Pehle dataset generate karein 'python dataset_generator.py'")
        return

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model initialization
    model = DigitCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nOCR training shuru ho rahi hai (Total Classes: {NUM_CLASSES}, Epochs: {epochs})...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Real-time progress bar loop
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", unit="batch", leave=True)
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Metrics update
            running_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            # Update tqdm real-time
            current_loss = running_loss / total_train
            current_acc = 100. * correct_train / total_train
            pbar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'acc': f"{current_acc:.2f}%"
            })
            
        # Check validation accuracy after each epoch
        test_loss, test_acc = check_accuracy(test_loader, model, criterion, device)
        print(f" -> Epoch Summary: Train Loss: {running_loss/total_train:.4f} | Train Acc: {100.*correct_train/total_train:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    print(f"\nTraining complete! Model ko '{MODEL_PATH}' par save kiya ja raha hai...")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model successfully save ho gaya.")

if __name__ == "__main__":
    train()
