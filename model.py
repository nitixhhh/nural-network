import torch
import torch.nn as nn
import torch.nn.functional as F

# Hum neural network class define kar rahe hain jo nn.Module se inherit karegi.
class DigitCNN(nn.Module):
    def __init__(self, num_classes=85): # Default 85 classes for General OCR (digits + alphabets + specials)
        super(DigitCNN, self).__init__()
        
        # Pehla Convolutional Layer (Conv1):
        # Input channel = 1 (Grayscale image)
        # Output channel = 16 (16 features extract karega)
        # Kernel size = 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        
        # Batch Normalization 1: Training ko stabilize aur generalisation ko improve karega
        self.bn1 = nn.BatchNorm2d(16)
        
        # Max Pooling Layer:
        # Kernel size = 2x2, Stride = 2
        # Input size: 26x26x16 -> Output size: 13x13x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dusra Convolutional Layer (Conv2):
        # Input channel = 16, Output channel = 32
        # Kernel size = 3x3
        # Max Pooling ke baad size hoga: 5x5x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(32)
        
        # Fully Connected Layer 1 (FC1):
        # Flattened input size = 32 channels * 5 height * 5 width = 800 features
        # Isse hum 128 hidden nodes mein map karenge
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        
        # Dropout: 20% connections ko drop karega training ke dauran overfitting rokne ke liye
        self.dropout = nn.Dropout(0.2)
        
        # Fully Connected Layer 2 (FC2 / Output Layer):
        # 128 features se num_classes (85) outputs denge (General OCR classes)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Forward pass definition:
        
        # 1. conv1 -> BatchNorm -> ReLU -> Max Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 2. conv2 -> BatchNorm -> ReLU -> Max Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 3. Flattening
        x = x.view(-1, 32 * 5 * 5)
        
        # 4. FC1 -> ReLU -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # 5. FC2 (Output logits)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Test karne ke liye ki model structural errors se free hai
    model = DigitCNN(num_classes=85)
    print("Robust OCR Model Architecture successfully created:")
    print(model)
    
    # Dummy input: 1 batch, 1 channel, 28x28 size
    dummy_input = torch.randn(1, 1, 28, 28)
    model.eval()
    output = model(dummy_input)
    print("\nDummy Input Shape:", dummy_input.shape)
    print("Output Shape:", output.shape)
    assert output.shape == (1, 85), "Output shape galat hai!"
    print("Assertion passed! Model properly output generate kar raha hai.")
