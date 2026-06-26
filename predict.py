import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from model import DigitCNN

# 85 OCR Classes definition (MUST EXACTLY MATCH dataset_generator.py)
digits = "0123456789"
uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowercase = "abcdefghijklmnopqrstuvwxyz"
specials = "@#%&+-*/=?!()[]{}<>;:.,"  # 23 characters
ALL_CLASSES = list(digits + uppercase + lowercase + specials)

# Auto-detecting model weights files:
# Agar OCR weights available hain toh wo load karega, nahi toh purana digits weight load karega.
if os.path.exists("ocr_model.pth"):
    MODEL_PATH = "ocr_model.pth"
    NUM_CLASSES = 85
    CLASS_MAPPING = ALL_CLASSES
    MODEL_TYPE = "General OCR (85 Classes)"
elif os.path.exists("digit_model.pth"):
    MODEL_PATH = "digit_model.pth"
    NUM_CLASSES = 10
    CLASS_MAPPING = [str(i) for i in range(10)]
    MODEL_TYPE = "Handwritten Digits (10 Classes)"
else:
    print("Error: Koi model weights (.pth) nahi mile! Pehle 'python train.py' run karein.")
    sys.exit(1)

def preprocess_image(image_path):
    """
    User image ko load karke thresholding, auto-cropping, centering aur normalization
    apply karne wala function.
    """
    if not os.path.exists(image_path):
        print(f"Error: Path '{image_path}' par koi image nahi mili!")
        sys.exit(1)
        
    # Image ko load aur Grayscale convert karte hain
    image = Image.open(image_path).convert('L')
    img_array = np.array(image)
    
    # Step 1: Robust Background Detection
    # Pure image ke pixel values ka median nikal kar background color detect karenge (most robust method).
    bg_color = np.median(img_array)
    
    # Step 2: Difference Image Creation
    # Background color se pixel difference nikalenge (background will become 0, digit will become bright).
    diff_array = np.abs(img_array.astype(np.int32) - bg_color)
    max_diff = diff_array.max()
    
    # Step 3: Dynamic Thresholding (Binarization)
    if max_diff < 15:
        print("[Preprocessing] Blank image detected.")
        bin_array = np.zeros_like(img_array, dtype=np.uint8)
    else:
        # Dynamic threshold: max_diff ka 20% ya minimum 15
        thresh = max(15, int(0.20 * max_diff))
        bin_array = np.where(diff_array > thresh, 255, 0).astype(np.uint8)
        
    # PIL format
    processed_image = Image.fromarray(bin_array)
    
    # Step 4: Auto-Cropping & Centering (MNIST/EMNIST standard matching)
    bbox = processed_image.getbbox()
    if bbox is not None:
        cropped = processed_image.crop(bbox)
        cw, ch = cropped.size
        
        # Aspect-ratio standard sizing to max 20x20
        max_dim = max(cw, ch)
        scale = 20.0 / max_dim
        new_w = max(1, int(cw * scale))
        new_h = max(1, int(ch * scale))
        
        cropped_resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Paste in center of 28x28
        centered_image = Image.new('L', (28, 28), color=0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        centered_image.paste(cropped_resized, (paste_x, paste_y))
        processed_image = centered_image
    else:
        processed_image = processed_image.resize((28, 28), Image.Resampling.LANCZOS)
        
    # Save preprocessed image for visual debugging
    debug_path = "debug_preprocessed.png"
    processed_image.save(debug_path)
    
    # Step 5: Normalize and Tensor Conversion
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img_tensor = transform(processed_image)
    img_tensor = img_tensor.unsqueeze(0)  # Shape: [1, 1, 28, 28]
    
    return img_tensor

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model ko load karenge dynamic class settings ke sath
    model = DigitCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print(f"\n[INFO] Running model: {MODEL_TYPE}")
    print(f"[INFO] Weights file: {MODEL_PATH}")
    print(f"Processing image: {image_path}...")
    
    img_tensor = preprocess_image(image_path).to(device)
    
    # Inference run karenge
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, dim=1)
        
    predicted_idx = predicted_class_idx.item()
    confidence_percentage = confidence.item() * 100
    
    # Class index ko character me map karenge
    predicted_char = CLASS_MAPPING[predicted_idx]
    
    print("\n" + "="*30)
    print(f"PREDICTED CHARACTER : {predicted_char}")
    print(f"CONFIDENCE          : {confidence_percentage:.2f}%")
    print("="*30)
    print("[Preprocessing] Preprocessed image saved to 'debug_preprocessed.png'\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Apni image ka path enter karein: ").strip()
        
    if not img_path:
        print("Error: Koi image path nahi diya gaya!")
        sys.exit(1)
        
    predict(img_path)
