import os
import sys
import numpy as np
from PIL import Image, ImageOps
import glob

# 85 OCR Classes (Must match dataset_generator.py exactly)
digits = "0123456789"
uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowercase = "abcdefghijklmnopqrstuvwxyz"
specials = "@#%&+-*/=?!()[]{}<>;:.,"
ALL_CLASSES = list(digits + uppercase + lowercase + specials)

# Mappings for safe folder names on case-insensitive filesystems (like Windows and Android exFAT)
special_mapping = {
    '@': 'at',
    '#': 'hash',
    '%': 'percent',
    '&': 'amp',
    '+': 'plus',
    '-': 'minus',
    '*': 'asterisk',
    '/': 'slash',
    '=': 'equal',
    '?': 'question',
    '!': 'excl',
    '(': 'paren_open',
    ')': 'paren_close',
    '[': 'bracket_open',
    ']': 'bracket_close',
    '{': 'brace_open',
    '}': 'brace_close',
    '<': 'lt',
    '>': 'gt',
    ';': 'semicolon',
    ':': 'colon',
    '.': 'dot',
    ',': 'comma'
}

# Create a bidirectional folder mapping
char_to_folder = {}
folder_to_char = {}

# 1. Digits: "0" to "9"
for d in digits:
    char_to_folder[d] = d
    folder_to_char[d] = d

# 2. Uppercase letters: A_upper, B_upper, ...
for u in uppercase:
    name = f"{u}_upper"
    char_to_folder[u] = name
    folder_to_char[name] = u
    folder_to_char[name.lower()] = u  # Case-insensitive fallback
    # Also support direct uppercase letter folder on case-sensitive filesystems
    folder_to_char[u] = u

# 3. Lowercase letters: a_lower, b_lower, ...
for l in lowercase:
    name = f"{l}_lower"
    char_to_folder[l] = name
    folder_to_char[name] = l
    folder_to_char[name.lower()] = l
    # Also support direct lowercase letter folder
    folder_to_char[l] = l

# 4. Specials mapping
for char, name in special_mapping.items():
    char_to_folder[char] = name
    folder_to_char[name] = char
    folder_to_char[name.lower()] = char
    # Also support direct character folder if allowed by OS
    folder_to_char[char] = char


def preprocess_custom_image(image_path):
    """
    User-provided image ko load karke thresholding, cropping aur centering apply
    karega taaki wo synthetic dataset ke structure (28x28) se align ho sake.
    """
    # Image load aur Grayscale convert
    image = Image.open(image_path).convert('L')
    img_array = np.array(image)
    
    # Background value median se identify karein
    bg_color = np.median(img_array)
    
    # Background subtraction (median difference)
    diff_array = np.abs(img_array.astype(np.int32) - bg_color)
    max_diff = diff_array.max()
    
    if max_diff < 15:
        # Blank image or no contrast
        bin_array = np.zeros_like(img_array, dtype=np.uint8)
    else:
        thresh = max(15, int(0.20 * max_diff))
        bin_array = np.where(diff_array > thresh, 255, 0).astype(np.uint8)
        
    processed_image = Image.fromarray(bin_array)
    
    # Auto-Cropping & Centering (MNIST style)
    bbox = processed_image.getbbox()
    if bbox is not None:
        cropped = processed_image.crop(bbox)
        cw, ch = cropped.size
        
        # Sizing inside 20x20 keeping aspect ratio
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
        
    return np.array(processed_image, dtype=np.uint8)


def main():
    data_dir = "custom_data"
    dataset_file = "ocr_dataset.npz"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        # Create a helper README for user
        with open(os.path.join(data_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write("Aap apne phone se laye gaye training data ko yahan add kar sakte hain.\n")
            f.write("Har character ke liye niche diye gaye name se ek subfolder banayein:\n\n")
            f.write("FOLDER NAMES MAPPING:\n")
            for char in ALL_CLASSES:
                f.write(f"  Character '{char}' -> Folder: {char_to_folder[char]}\n")
        print(f"[INFO] '{data_dir}' folder create kiya gaya hai. Subfolders aur images daal kar fir se run karein.")
        print(f"Details '{data_dir}/README.txt' me save kar di gayi hain.")
        return

    # Check existing dataset
    if os.path.exists(dataset_file):
        print(f"[INFO] Existing dataset '{dataset_file}' ko load kiya ja raha hai...")
        existing_data = np.load(dataset_file)
        existing_images = existing_data["images"]
        existing_labels = existing_data["labels"]
        print(f"Original images count: {len(existing_images)}")
    else:
        print(f"[WARNING] '{dataset_file}' nahi mila! Naya dataset file banaya jayega.")
        existing_images = np.empty((0, 28, 28), dtype=np.uint8)
        existing_labels = np.empty((0,), dtype=np.int64)

    new_images = []
    new_labels = []
    summary_added = {}

    # Scan for folders inside custom_data
    subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    for folder_name in subfolders:
        # Check mapping
        char = folder_to_char.get(folder_name) or folder_to_char.get(folder_name.lower())
        if char is None:
            print(f"[SKIP] Unknown folder structure: '{folder_name}' (OCR class list me nahi mila)")
            continue
            
        class_idx = ALL_CLASSES.index(char)
        folder_path = os.path.join(data_dir, folder_name)
        
        # Supported image formats
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
        if len(image_paths) == 0:
            continue
            
        print(f"[PROCESSING] Folder '{folder_name}' -> Class '{char}' | Total Images: {len(image_paths)}")
        
        added_for_char = 0
        for img_path in image_paths:
            try:
                preprocessed = preprocess_custom_image(img_path)
                new_images.append(preprocessed)
                new_labels.append(class_idx)
                added_for_char += 1
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                
        if added_for_char > 0:
            summary_added[char] = added_for_char

    if len(new_images) > 0:
        new_images_arr = np.array(new_images, dtype=np.uint8)
        new_labels_arr = np.array(new_labels, dtype=np.int64)
        
        # Concatenate with existing
        if len(existing_images) > 0:
            combined_images = np.concatenate((existing_images, new_images_arr), axis=0)
            combined_labels = np.concatenate((existing_labels, new_labels_arr), axis=0)
        else:
            combined_images = new_images_arr
            combined_labels = new_labels_arr
            
        # Save updated dataset
        np.savez_compressed(dataset_file, images=combined_images, labels=combined_labels)
        
        print("\n" + "="*40)
        print("[SUCCESS] Dataset updated successfully!")
        print(f"Added custom samples details:")
        for char, count in summary_added.items():
            print(f"  Character '{char}': {count} images added")
        print("-"*40)
        print(f"Updated dataset size: {combined_images.shape[0]} total images")
        print("="*40 + "\n")
        print("Aap training start karne ke liye ye command chala sakte hain:")
        print("  python train.py --force\n")
    else:
        print("\n[INFO] Koi naya custom training data (images) nahi mila. Make sure images folder me sahi format (.png, .jpg) me hain.")
        print("Folder structure details ke liye 'custom_data/README.txt' read karein.\n")


if __name__ == "__main__":
    main()
