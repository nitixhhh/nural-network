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


def get_binarized_array(image):
    """
    User image ko load karke difference-based dynamic binarization return karega.
    """
    img_array = np.array(image.convert('L'))
    bg_color = np.median(img_array)
    diff_array = np.abs(img_array.astype(np.int32) - bg_color)
    max_diff = diff_array.max()
    
    if max_diff < 15:
        return np.zeros_like(img_array, dtype=np.uint8)
    
    thresh = max(15, int(0.20 * max_diff))
    return np.where(diff_array > thresh, 255, 0).astype(np.uint8)


def center_and_scale(cropped_pil):
    """
    Crop kiye character ko [20x20] me scale aur [28x28] grid me center karega.
    """
    cw, ch = cropped_pil.size
    max_dim = max(cw, ch)
    scale = 20.0 / max_dim
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    
    cropped_resized = cropped_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    centered_image = Image.new('L', (28, 28), color=0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    centered_image.paste(cropped_resized, (paste_x, paste_y))
    return centered_image


def segment_characters(bin_array):
    """
    Pure Python/NumPy vertical projection profiling to segment multiple characters
    written horizontally. Returns a list of 28x28 preprocessed PIL images.
    """
    h, w = bin_array.shape
    col_sums = np.sum(bin_array, axis=0)
    active_cols = col_sums > 0
    
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, active in enumerate(active_cols):
        if active and not in_segment:
            start_idx = i
            in_segment = True
        elif not active and in_segment:
            # Ignore tiny noise columns (e.g. less than 2 pixels wide)
            if i - start_idx >= 2:
                segments.append((start_idx, i))
            in_segment = False
            
    if in_segment:
        if w - start_idx >= 2:
            segments.append((start_idx, w))
            
    # Agar koi character segment nahi mila, toh fallback to full image
    if len(segments) == 0:
        pil_img = Image.fromarray(bin_array)
        bbox = pil_img.getbbox()
        if bbox is not None:
            return [center_and_scale(pil_img.crop(bbox))]
        return [center_and_scale(pil_img)]
        
    cropped_chars = []
    for start, end in segments:
        char_slice = bin_array[:, start:end]
        row_sums = np.sum(char_slice, axis=1)
        active_rows = np.where(row_sums > 0)[0]
        if len(active_rows) == 0:
            continue
        top, bottom = active_rows[0], active_rows[-1] + 1
        
        char_crop = char_slice[top:bottom, :]
        cropped_chars.append(center_and_scale(Image.fromarray(char_crop)))
        
    return cropped_chars


def get_character_details(char):
    """
    Returns (Type, ASCII) for a given character.
    """
    if char in digits:
        return "Digit (Number)", ord(char)
    elif char in uppercase:
        return "Uppercase Alphabet Letter", ord(char)
    elif char in lowercase:
        return "Lowercase Alphabet Letter", ord(char)
    elif char in specials:
        special_names = {
            '@': 'At Sign (E-mail)',
            '#': 'Hash / Octothorpe',
            '%': 'Percent Sign',
            '&': 'Ampersand (And)',
            '+': 'Plus Sign (Addition)',
            '-': 'Minus Sign / Hyphen',
            '*': 'Asterisk (Multiplication)',
            '/': 'Forward Slash (Division)',
            '=': 'Equal Sign',
            '?': 'Question Mark',
            '!': 'Exclamation Mark',
            '(': 'Open Parenthesis',
            ')': 'Close Parenthesis',
            '[': 'Open Bracket',
            ']': 'Close Bracket',
            '{': 'Open Brace',
            '}': 'Close Brace',
            '<': 'Less Than Sign',
            '>': 'Greater Than Sign',
            ';': 'Semicolon',
            ':': 'Colon',
            '.': 'Period / Dot',
            ',': 'Comma'
        }
        return f"Special Symbol ({special_names.get(char, 'Character')})", ord(char)
    return "Unknown Character", ord(char)


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
        
        # Preprocess & Segment
        bin_array = get_binarized_array(img)
        cropped_chars = segment_characters(bin_array)
        print(f"[Segmentation] Detected {len(cropped_chars)} individual characters.")
        
        predicted_chars_list = []
        confidences_list = []
        
        # Run inference for each segmented character
        for idx, char_img in enumerate(cropped_chars):
            img_tensor = transform(char_img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class_idx = torch.max(probabilities, dim=1)
                
            predicted_idx = predicted_class_idx.item()
            predicted_chars_list.append(CLASS_MAPPING[predicted_idx])
            confidences_list.append(confidence.item() * 100)
            
        # Combine characters
        predicted_string = "".join(predicted_chars_list)
        average_confidence = sum(confidences_list) / len(confidences_list)
        
        # Save tiled debug image (all cropped characters side-by-side)
        tiled_image = Image.new('L', (28 * len(cropped_chars), 28), color=0)
        for idx, char_img in enumerate(cropped_chars):
            tiled_image.paste(char_img, (28 * idx, 0))
            
        debug_path = os.path.join("static", "debug_preprocessed.png")
        tiled_image.save(debug_path)
        print("[Preprocessing] Tiled preprocessed debug image saved to 'static/debug_preprocessed.png'")
        
        # Details metadata parsing
        if len(predicted_string) == 1:
            char_type, ascii_val = get_character_details(predicted_string)
        else:
            char_types_set = set()
            for c in predicted_string:
                t, _ = get_character_details(c)
                char_types_set.add(t.split(" (")[0].split(" ")[0])
            char_type = "Multi-Character Text (" + " + ".join(sorted(char_types_set)) + ")"
            ascii_val = ", ".join(str(ord(c)) for c in predicted_string)
            
        # Print output to terminal console (exact box format user loves)
        print("\n" + "="*30)
        print(f"PREDICTED STRING    : {predicted_string}")
        print(f"AVG CONFIDENCE      : {average_confidence:.2f}%")
        print(f"TYPE                : {char_type}")
        print(f"ASCII VALUES        : {ascii_val}")
        print("="*30 + "\n")
        
        return jsonify({
            'character': predicted_string,
            'confidence': round(average_confidence, 2),
            'char_type': char_type,
            'ascii': ascii_val,
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
