import numpy as np
from PIL import Image, ImageFilter, ImageOps

def try_advanced_segmentation(image_path, out_prefix):
    # Load original image
    image = Image.open(image_path)
    
    # 1. Grayscale
    gray = image.convert('L')
    
    # 2. Try different Edge Detectors
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges.save(f"{out_prefix}_edges.png")
    
    # 3. Try Adaptive Local Thresholding or large Median Filter
    # Let's try to remove high frequency lines by applying a strong Median Filter (size 9)
    median_9 = gray.filter(ImageFilter.MedianFilter(size=9))
    median_9.save(f"{out_prefix}_median9.png")
    
    # 4. Try MinFilter (Erosion) and MaxFilter (Dilation)
    # A MinFilter (Erosion) will suppress thin bright lines.
    # A MaxFilter (Dilation) will suppress thin dark lines.
    min_filter = gray.filter(ImageFilter.MinFilter(size=5))
    min_filter.save(f"{out_prefix}_min5.png")
    
    max_filter = gray.filter(ImageFilter.MaxFilter(size=5))
    max_filter.save(f"{out_prefix}_max5.png")

if __name__ == "__main__":
    try_advanced_segmentation("gpt0.png", "temp_gpt0")
    try_advanced_segmentation("gpt5.png", "temp_gpt5")
    print("Advanced filtering tests completed. Images saved.")
