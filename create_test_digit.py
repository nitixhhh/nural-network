import os
from PIL import Image, ImageDraw

def create_sample_digit():
    # 100x100 dimensions ki ek white image create karenge (background = 255)
    # MNIST has dark background, but hum predict.py ki auto-inversion feature 
    # check karne ke liye white background image draw karenge (black digit ke sath).
    img = Image.new('L', (100, 100), color=255)
    draw = ImageDraw.Draw(img)
    
    # Digit '7' draw karenge using lines
    # Horizontal line (top of 7)
    draw.line([(25, 25), (75, 25)], fill=0, width=8)
    # Diagonal line (stem of 7)
    draw.line([(75, 25), (45, 80)], fill=0, width=8)
    
    # Image save karte hain
    filename = "test_digit_7.png"
    img.save(filename)
    print(f"Sample digit image successfully save ho gayi: {filename}")

if __name__ == "__main__":
    create_sample_digit()
