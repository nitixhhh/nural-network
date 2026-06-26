import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 85 Classes list
digits = "0123456789"
uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowercase = "abcdefghijklmnopqrstuvwxyz"
specials = "@#%&+-*/=?!()[]{}<>;:.,"  # 23 characters
ALL_CLASSES = list(digits + uppercase + lowercase + specials)
NUM_CLASSES = len(ALL_CLASSES)
SAMPLES_PER_CLASS = 500  # 300 se badha kar 500 kiya taaki 42,500 total diverse samples generate ho sakein.

def get_system_fonts():
    """
    Cross-platform font paths return karega (Windows, Linux, macOS, Android Termux proot).
    """
    fonts = []
    if os.name == 'nt':  # Windows OS
        font_dir = os.environ.get('WINDIR', 'C:\\Windows') + '\\Fonts'
        standard_fonts = ['arial.ttf', 'times.ttf', 'cour.ttf', 'calibri.ttf', 'segoeui.ttf', 'georgia.ttf', 'consola.ttf']
        for f in standard_fonts:
            p = os.path.join(font_dir, f)
            if os.path.exists(p):
                fonts.append(p)
    else:  # Linux / Android Termux proot / macOS
        # Common Linux/Android paths (Apt packages like fonts-dejavu, fonts-liberation install karne par aate hain)
        linux_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        ]
        for p in linux_paths:
            if os.path.exists(p):
                fonts.append(p)
                
    return fonts

SYSTEM_FONTS = get_system_fonts()

def elastic_warp(img_arr):
    """
    Standard linear computer fonts ko dynamic hand-wiggles (handwriting fluctuations)
    me transform karne ke liye numpy-based fast warping function.
    """
    h, w = img_arr.shape
    # Random amplitude (stroke wobble intensity) and period (wobble frequency)
    amplitude = random.uniform(0.6, 2.2)
    period = random.uniform(9.0, 18.0)
    
    y_indices, x_indices = np.indices((h, w))
    dx = amplitude * np.sin(2 * np.pi * y_indices / period)
    dy = amplitude * np.cos(2 * np.pi * x_indices / period)
    
    # Clip indices to prevent out of bounds
    new_x = np.clip(x_indices + dx, 0, w - 1).astype(np.int32)
    new_y = np.clip(y_indices + dy, 0, h - 1).astype(np.int32)
    
    return img_arr[new_y, new_x]

def get_random_font():
    if len(SYSTEM_FONTS) > 0:
        random.shuffle(SYSTEM_FONTS)
        for font_path in SYSTEM_FONTS:
            try:
                size = random.randint(24, 38)
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def generate_ocr_dataset():
    print(f"OCR Dataset generator starting... Total Classes: {NUM_CLASSES}")
    print(f"Detected Fonts count: {len(SYSTEM_FONTS)}")
    print(f"Generating {SAMPLES_PER_CLASS} samples per class with handwriting slants & stroke thickness variation...")
    
    images_list = []
    labels_list = []
    
    for class_idx, char in enumerate(ALL_CLASSES):
        if (class_idx + 1) % 10 == 0 or (class_idx + 1) == NUM_CLASSES:
            print(f"Processing classes... Progress: {class_idx + 1}/{NUM_CLASSES} ('{char}')")
            
        for _ in range(SAMPLES_PER_CLASS):
            # 1. Canvas creation
            temp_img = Image.new('L', (64, 64), color=0)
            draw = ImageDraw.Draw(temp_img)
            
            font = get_random_font()
            
            # Center alignment check
            bbox = draw.textbbox((0, 0), char, font=font)
            w_txt = bbox[2] - bbox[0]
            h_txt = bbox[3] - bbox[1]
            
            shift_x = random.randint(-4, 4)
            shift_y = random.randint(-4, 4)
            pos_x = (64 - w_txt) // 2 - bbox[0] + shift_x
            pos_y = (64 - h_txt) // 2 - bbox[1] + shift_y
            
            draw.text((pos_x, pos_y), char, fill=255, font=font)
            
            # [UPGRADE: Slant / Shear transformation]
            # Log hand-drawing ke waqt aksar thoda slanted (slant) likhte hain.
            # Hum image par random horizontal shear apply karenge.
            shear_factor = random.uniform(-0.35, 0.35)  # Left-Right skew slanting badha diya gaya hai robust matching ke liye
            temp_img = temp_img.transform(
                (64, 64), 
                Image.Transform.AFFINE, 
                (1, shear_factor, 0, 0, 1, 0), 
                resample=Image.Resampling.BILINEAR
            )
            
            # [UPGRADE: Handwriting Elastic Warp]
            # Straight edges ko curved aur wavy hand-written aesthetics dene ke liye
            temp_arr = np.array(temp_img)
            warped_arr = elastic_warp(temp_arr)
            temp_img = Image.fromarray(warped_arr)
            
            # 2. Random Rotation
            angle = random.uniform(-20, 20)  # Rotation range -15 to +15 se bada kar -20 to +20 kiya
            temp_img = temp_img.rotate(angle, Image.Resampling.BILINEAR)
            
            # [UPGRADE: Stroke Thickness Variation]
            # Kuch log pen se patla likhte hain aur kuch marker/thick brush se.
            # MaxFilter stroke ko thick karta hai (dilation), MinFilter thin karta hai (erosion).
            thickness_rand = random.random()
            if thickness_rand < 0.3:
                # Thick stroke simulation
                temp_img = temp_img.filter(ImageFilter.MaxFilter(3))
            elif thickness_rand < 0.6:
                # Thin stroke simulation
                temp_img = temp_img.filter(ImageFilter.MinFilter(3))
            
            # 3. Auto-cropping & Centering (MNIST Matching)
            bbox_crop = temp_img.getbbox()
            if bbox_crop is not None:
                cropped = temp_img.crop(bbox_crop)
                cw, ch = cropped.size
                max_dim = max(cw, ch)
                
                if max_dim == 0:
                    max_dim = 1
                    
                scale = 20.0 / max_dim
                new_w = max(1, int(cw * scale))
                new_h = max(1, int(ch * scale))
                
                cropped_resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                centered_image = Image.new('L', (28, 28), color=0)
                paste_x = (28 - new_w) // 2
                paste_y = (28 - new_h) // 2
                centered_image.paste(cropped_resized, (paste_x, paste_y))
            else:
                centered_image = temp_img.resize((28, 28), Image.Resampling.LANCZOS)
                
            # 4. Add minor Gaussian Noise
            img_arr = np.array(centered_image)
            noise = np.random.normal(0, 10, img_arr.shape).astype(np.int16)
            img_arr_noisy = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            images_list.append(img_arr_noisy)
            labels_list.append(class_idx)
            
    # Save as compressed npz file
    images_array = np.array(images_list, dtype=np.uint8)
    labels_array = np.array(labels_list, dtype=np.int64)
    
    dataset_file = "ocr_dataset.npz"
    np.savez_compressed(dataset_file, images=images_array, labels=labels_array)
    print(f"\n[SUCCESS] Dataset generated and saved to '{dataset_file}'!")
    print(f"Images shape: {images_array.shape} | Labels shape: {labels_array.shape}")

if __name__ == "__main__":
    generate_ocr_dataset()
