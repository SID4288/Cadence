# new/model.py — Specialized 2-Channel Residual Audio CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2D(nn.Module):
    """
    Highly stable 2D Residual block with Batch Normalization, ReLU, and Dropout.
    Designed specifically for Time-Frequency Spectrogram processing.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # Shortcut connection for residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FolkMusicCQTResNet(nn.Module):
    """
    SOTA 2-Channel Custom ResNet tailored for CQT music representations.
    Input Shape: (Batch, 2, N_BINS=84, N_FRAMES=~1292)
    Channel 0: Harmonic component
    Channel 1: Percussive component
    """
    def __init__(self, num_classes=6, dropout_prob=0.2):
        super().__init__()
        
        # Initial representation convolution
        self.conv_init = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(32)
        
        # High-performance residual hierarchy
        self.layer1 = ResBlock2D(32, 64, stride=1, dropout_prob=dropout_prob)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # (84, 1292) -> (42, 646)
        
        self.layer2 = ResBlock2D(64, 128, stride=1, dropout_prob=dropout_prob)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # (42, 646) -> (21, 323)
        
        self.layer3 = ResBlock2D(128, 256, stride=1, dropout_prob=dropout_prob)
        # We pool 1 in frequency, 2 in time since time is significantly larger
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))  # (21, 323) -> (21, 161)
        
        self.layer4 = ResBlock2D(256, 256, stride=1, dropout_prob=dropout_prob)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))  # (21, 161) -> (10, 80)
        
        self.layer5 = ResBlock2D(256, 512, stride=1, dropout_prob=dropout_prob)
        
        # Global Average Pooling to flatten spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Initial features
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        # Block propagation
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
        x = self.layer5(x)
        
        # Spatial consolidation
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Classification prediction
        logits = self.fc(x)
        return logits


if __name__ == "__main__":
    # Performance profile test
    model = FolkMusicCQTResNet(num_classes=6)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=== MODEL PERFORMANCE PROFILE ===")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Test with standard CQT dimensions (Batch=4, Channels=2, Bins=84, Frames=1292)
    dummy_input = torch.randn(4, 2, 84, 1292)
    output = model(dummy_input)
    print(f"Input Shape:          {dummy_input.shape}")
    print(f"Output Shape:         {output.shape}")
    print("=================================")
