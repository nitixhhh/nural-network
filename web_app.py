import os
import io
import sys
import base64
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from model import DigitCNN

app = Flask(__name__)

# Create static directory for visual debug images
os.makedirs("static", exist_ok=True)

# 85 OCR Classes definition (MUST EXACTLY MATCH dataset_generator.py)
digits = "0123456789"
uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowercase = "abcdefghijklmnopqrstuvwxyz"
specials = "@#%&+-*/=?!()[]{}<>;:.,"  # 23 characters
ALL_CLASSES = list(digits + uppercase + lowercase + specials)

# Auto-detecting model weights files:
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

# Device Configuration and Model Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Normalization transform for PyTorch input matching
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def handle_transparency_and_convert(image):
    """
    If drawn on a transparent canvas, converts transparent pixels to white background,
    making sure the drawn paths remain highly visible.
    """
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # Create a white background canvas
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        bg.paste(image, (0, 0), image)
        return bg.convert('RGB')
    return image


def preprocess_image_from_pil(image):
    """
    Same robust preprocessing logic as predict.py (inversion, cropping, centering)
    """
    img_array = np.array(image.convert('L'))
    
    # Step 1: Robust Background Detection
    bg_color = np.median(img_array)
    
    # Step 2: Difference Image Creation
    diff_array = np.abs(img_array.astype(np.int32) - bg_color)
    max_diff = diff_array.max()
    
    # Step 3: Dynamic Thresholding (Binarization)
    if max_diff < 15:
        bin_array = np.zeros_like(img_array, dtype=np.uint8)
    else:
        thresh = max(15, int(0.20 * max_diff))
        bin_array = np.where(diff_array > thresh, 255, 0).astype(np.uint8)
        
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
        
    # Save preprocessed image for visual debugging (loaded by UI)
    debug_path = os.path.join("static", "debug_preprocessed.png")
    processed_image.save(debug_path)
    
    return processed_image


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Handle transparency & background
        img = handle_transparency_and_convert(img)
        
        # Log to terminal console in predict.py format
        print(f"\n[INFO] Running model: {MODEL_TYPE}")
        print(f"[INFO] Weights file: {MODEL_PATH}")
        print("[Web API] Received image for prediction...")
        
        # Preprocess
        preprocessed_img = preprocess_image_from_pil(img)
        print("[Preprocessing] Preprocessed debug image saved to 'static/debug_preprocessed.png'")
        
        # Run inference
        img_tensor = transform(preprocessed_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)
            
        predicted_idx = predicted_class_idx.item()
        confidence_percentage = confidence.item() * 100
        predicted_char = CLASS_MAPPING[predicted_idx]
        
        # Print output to terminal console (exact box format user loves)
        print("\n" + "="*30)
        print(f"PREDICTED CHARACTER : {predicted_char}")
        print(f"CONFIDENCE          : {confidence_percentage:.2f}%")
        print("="*30 + "\n")
        
        return jsonify({
            'character': predicted_char,
            'confidence': round(confidence_percentage, 2),
            'debug_image': '/static/debug_preprocessed.png'
        })
        
    except Exception as e:
        print(f"[ERROR] Exception during prediction: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    print("\n" + "*"*50)
    print("      AI GENERAL OCR WEB SERVER STARTING...")
    print(f"      Model loaded: {MODEL_TYPE}")
    print(f"      Weights file: {MODEL_PATH}")
    print(f"      Running on device: {device}")
    print("*"*50)
    print("\nOpen Chrome on your phone or PC at:")
    print("   ---> http://localhost:5000  (Same device)")
    print("   ---> http://<YOUR_IP>:5000   (Mobile from same Wi-Fi network)\n")
    
    # host='0.0.0.0' binds to all network interfaces, allowing phone connectivity
    app.run(host='0.0.0.0', port=5000, debug=False)
