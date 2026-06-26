# PyTorch General OCR Engine & Digit Classifier

A complete, beginner-friendly Convolutional Neural Network (CNN) project built in PyTorch. The system supports two modes of operation:
1. **Digit Recognition Mode (10 Classes)**: Trained on the classic MNIST dataset to classify digits `0-9`.
2. **General OCR Mode (85 Classes)**: Trained on a dynamically synthesized, font-distorted dataset to classify:
   - Digits `0-9` (10 classes)
   - Uppercase letters `A-Z` (26 classes)
   - Lowercase letters `a-z` (26 classes)
   - Special characters `@`, `#`, `%`, `&`, `+`, `-`, `*`, `/`, `=`, `?`, `!`, `$`, `(`, `)`, `[`, `]`, `{`, `}`, `<`, `>`, `:`, `;`, `,`, `.` (23 classes)

---

## Features

- **Double-Model Capability**: Supports both `digit_model.pth` and `ocr_model.pth` with automatic model selection based on weight file availability.
- **On-the-Fly Font Dataset Synthesizer**: Uses standard system fonts with random rotation, translation, scaling, and noise to generate robust OCR training datasets (`ocr_dataset.npz`) instantly.
- **10x Vectorized Training Performance**: Dataset pre-processed directly to normalized PyTorch tensors, reducing training times to under 40 seconds per epoch on CPU.
- **Robust Inference Processing**: 
  - **Median-Based Background Subtraction**: Wipes out solid backgrounds of any color (e.g. Red-on-Green) automatically.
  - **Dynamic Binarization & Centering**: Automatically crops the character bounding box, resizes it to `20x20` preserving aspect ratio, and centers it in a standard `28x28` black grid.
- **Beautiful Terminal Dashboard**: Integrated with `tqdm` to show progress bars, loss, and accuracy in real-time during training and validation.

---

## Directory Structure

```text
nural-network/
│
├── model.py                # Parameterized CNN architecture
├── train.py                # OCR & Digit training logic with progress bar
├── predict.py              # Advanced preprocessing and inference script
├── dataset_generator.py    # Synthesizes 25,500 character images using system fonts
├── requirements.txt        # Project dependencies (includes tqdm)
├── .vscode/
│   └── settings.json       # Editor configurations for local virtual environment
└── README.md               # User documentation
```

---

## Setup & Installation

### 1. Initialize Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run OCR Mode (85 Classes)

### Step 1: Synthesize the Dataset
Generate 25,500 distorted character images from system fonts:
```bash
python dataset_generator.py
```
*Outputs: `ocr_dataset.npz`*

### Step 2: Train the Model
Train the 85-class OCR CNN model:
```bash
python train.py
```
*Outputs: `ocr_model.pth`*

### Step 3: Run Prediction
Test on any custom image:
```bash
python predict.py path/to/your/image.png
```

---

## How to Run Digit Mode (10 Classes)

To run the classic 10-digit classifier:
1. Ensure `ocr_model.pth` is deleted or renamed, and `digit_model.pth` is active.
2. Run `train.py` (which will download the standard MNIST dataset if not present):
   ```bash
   python train.py
   ```
3. Test your custom digits:
   ```bash
   python predict.py test_digit_7.png
   ```

*Note: `predict.py` automatically detects which model is present in the workspace (`ocr_model.pth` or `digit_model.pth`) and adjusts classes mapping accordingly!*

---

## Web UI App (Canvas & Mobile Upload)

You can run a local web server to draw characters with your mouse/touch, or capture photos from your phone camera and upload images from your gallery.

### Launching the Web App:
```bash
python web_app.py
```
* Binds to port `5000` by default.
* Open your browser at:
  * On the same device: `http://localhost:5000`
  * From another device (e.g. phone) on the same Wi-Fi: `http://<YOUR_PC_IP>:5000`
* Draws are dynamically preprocessed and fed into the AI model, displaying results both on the web screen and logging full details to the Termux/PC console.

---

## Adding Custom Training Data

If you want to train the model on your own handwriting styles:

1. Create a `custom_data/` folder (created automatically on first run of `add_custom_data.py`).
2. Inside `custom_data/`, create subfolders named after the target characters. To prevent case-sensitivity and OS issues (especially on Windows & Android), use the following naming mapping:
   * Digits `0-9` -> `0`, `1`, ..., `9`
   * Uppercase `A-Z` -> `A_upper`, `B_upper`, ..., `Z_upper`
   * Lowercase `a-z` -> `a_lower`, `b_lower`, ..., `z_lower`
   * Special Characters -> `at` (`@`), `hash` (`#`), `percent` (`%`), `amp` (`&`), `plus` (`+`), `minus` (`-`), `asterisk` (`*`), `slash` (`/`), `equal` (`=`), `question` (`?`), `excl` (`!`), `paren_open` (`(`), `paren_close` (`)`), etc.
3. Save your handwritten drawings/images (e.g. `.png` or `.jpg`) in their corresponding character subfolders.
4. Run the data loader script to preprocess and append this data to the dataset:
   ```bash
   python add_custom_data.py
   ```
5. Force-retrain the model on the updated dataset:
   ```bash
   python train.py --force --epochs 6
   ```

---

## Cleaning Temporary Files

To clean up debug PNG images that accumulate in the project root:
```bash
python clean_project.py
```

---

## Running on Android (Termux Setup)

Since PyTorch can be tricky to compile on native Android, the recommended and most robust way is to run a **PRoot Ubuntu** environment inside Termux.

### Step 1: Install Termux
Install Termux via [F-Droid](https://f-droid.org/) (do not use the obsolete Google Play version).

### Step 2: Grant Storage Permission
Open Termux and run:
```bash
termux-setup-storage
```
*This mounts your phone's internal storage inside Termux at `~/storage/shared/`.*

### Step 3: Setup PRoot Ubuntu
Run the following commands in Termux:
```bash
pkg update && pkg upgrade -y
pkg install proot-distro -y
proot-distro install ubuntu
proot-distro login ubuntu
```

### Step 4: Install Dependencies inside Ubuntu
Inside the logged-in Ubuntu session, run:
```bash
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git
```

### Step 5: Transfer / Clone Project & Run
Clone the repository or copy files into your Ubuntu folder, activate virtual environment, and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pillow tqdm flask
```

Run the web application to draw or capture photos:
```bash
python web_app.py
```
Open your mobile browser at `http://localhost:5000` to start drawing and predicting!

