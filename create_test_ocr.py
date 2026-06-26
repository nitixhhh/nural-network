import os
from PIL import Image, ImageDraw, ImageFont

def create_sample_char(char, filename):
    # 100x100 white image
    img = Image.new('L', (100, 100), color=255)
    draw = ImageDraw.Draw(img)
    
    # Windows standard font path check
    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 60)
    else:
        font = ImageFont.load_default()
        
    # Get size
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Center text
    pos_x = (100 - w) // 2 - bbox[0]
    pos_y = (100 - h) // 2 - bbox[1]
    
    # Draw character in black (0)
    draw.text((pos_x, pos_y), char, fill=0, font=font)
    img.save(filename)
    print(f"Generated OCR test file: {filename} ('{char}')")

if __name__ == "__main__":
    create_sample_char("A", "test_char_A.png")
    create_sample_char("g", "test_char_g.png")
    create_sample_char("@", "test_char_at.png")
    create_sample_char("+", "test_char_plus.png")
