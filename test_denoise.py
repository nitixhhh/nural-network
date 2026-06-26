import numpy as np
from PIL import Image, ImageFilter

def keep_largest_components(bin_array, max_components=2, min_area=30):
    """
    Connected components analysis using a simple BFS in Python.
    Keepts only the largest components (which are the digits) and removes small background noise.
    """
    h, w = bin_array.shape
    visited = np.zeros_like(bin_array, dtype=bool)
    components = []
    
    for y in range(h):
        for x in range(w):
            if bin_array[y, x] == 255 and not visited[y, x]:
                # Start BFS
                queue = [(y, x)]
                visited[y, x] = True
                comp_pixels = []
                head = 0
                
                while head < len(queue):
                    cy, cx = queue[head]
                    head += 1
                    comp_pixels.append((cy, cx))
                    
                    # 8-connectivity check
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if bin_array[ny, nx] == 255 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))
                                    
                if len(comp_pixels) >= min_area:
                    components.append(comp_pixels)
                    
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    # Create new clean binary array
    clean_array = np.zeros_like(bin_array)
    
    # Keep only the top 'max_components' largest components
    for comp in components[:max_components]:
        for y, x in comp:
            clean_array[y, x] = 255
            
    return clean_array

def test_denoise(image_path, out_path):
    image = Image.open(image_path).convert('L')
    
    # Optional median filter to remove initial high-frequency salt/pepper noise
    image_filtered = image.filter(ImageFilter.MedianFilter(size=3))
    
    img_array = np.array(image_filtered)
    bg_color = np.median(img_array)
    
    # Difference image
    diff_array = np.abs(img_array.astype(np.int32) - bg_color)
    max_diff = diff_array.max()
    
    if max_diff < 15:
        bin_array = np.zeros_like(img_array, dtype=np.uint8)
    else:
        # Dynamic threshold
        thresh = max(15, int(0.20 * max_diff))
        bin_array = np.where(diff_array > thresh, 255, 0).astype(np.uint8)
        
    # Apply Connected Components filter to remove scattered noise!
    # We keep the top 1 or 2 largest components.
    bin_array = keep_largest_components(bin_array, max_components=2, min_area=30)
    
    processed_image = Image.fromarray(bin_array)
    
    # Centering & scaling
    bbox = processed_image.getbbox()
    if bbox is not None:
        cropped = processed_image.crop(bbox)
        cw, ch = cropped.size
        max_dim = max(cw, ch)
        scale = 20.0 / max_dim
        new_w = max(1, int(cw * scale))
        new_h = max(1, int(ch * scale))
        cropped_resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        centered_image = Image.new('L', (28, 28), color=0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        centered_image.paste(cropped_resized, (paste_x, paste_y))
        processed_image = centered_image
    else:
        processed_image = processed_image.resize((28, 28), Image.Resampling.LANCZOS)
        
    processed_image.save(out_path)
    print(f"Saved robust denoise test to: {out_path}")

if __name__ == "__main__":
    test_denoise("gpt0.png", "denoise_gpt0.png")
    test_denoise("gpt5.png", "denoise_gpt5.png")
