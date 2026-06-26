import os
import glob

def clean_png_files():
    print("[INFO] Cleaning temporary PNG files from the project root...")
    # Find all PNG files in the current directory
    png_files = glob.glob("*.png")
    
    deleted_count = 0
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            
    print(f"[SUCCESS] Cleaned up {deleted_count} temporary PNG files.\n")

if __name__ == "__main__":
    clean_png_files()
