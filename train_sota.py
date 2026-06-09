# new/train_sota.py — SOTA Training Pipeline with SpecAugment
import inspect
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time

# Anchor CWD to the `new/` directory so relative paths like `models/`
# and `results/` always resolve inside `new/`.
NEW_ROOT = Path(__file__).resolve().parent
os.chdir(str(NEW_ROOT))

sys.path.append(str(NEW_ROOT))
from model import FolkMusicCQTResNet
from dataset import create_sota_dataloaders
from utils import GENRES, print_settings

class SpecAugment:
    """
    SpecAugment implementation for audio spectrograms.
    Applies frequency and time masking to augment training data.
    """
    def __init__(self, freq_mask_max=15, time_mask_max=40, num_freq_masks=2, num_time_masks=2):
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, specrogram):
        """
        Apply SpecAugment to a spectrogram of shape (2, freq_bins, time_frames)
        """
        specrogram = specrogram.clone()

        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            if specrogram.shape[1] > self.freq_mask_max:
                f = np.random.randint(0, self.freq_mask_max)
                f0 = np.random.randint(0, specrogram.shape[1] - f)
                specrogram[:, f0:f0+f, :] = 0

        # Apply time masking
        for _ in range(self.num_time_masks):
            if specrogram.shape[2] > self.time_mask_max:
                t = np.random.randint(0, self.time_mask_max)
                t0 = np.random.randint(0, specrogram.shape[2] - t)
                specrogram[:, :, t0:t0+t] = 0

        return specrogram

def train_epoch(model, train_loader, optimizer, criterion, device, spec_augment=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)

        # Apply SpecAugment during training only
        if spec_augment is not None:
            features = spec_augment(features)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}', flush=True)

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=GENRES)
    cm = confusion_matrix(all_labels, all_preds)

    epoch_loss = running_loss / len(test_loader)
    return epoch_loss, accuracy, report, cm

def save_model(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f'Model saved to {filepath}', flush=True)

def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GENRES, yticklabels=GENRES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f'Confusion matrix saved to {save_path}')

def main():
    # Initialize settings
    print_settings()

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    batch_size = 16
    num_epochs = 25
    learning_rate = 0.001
    weight_decay = 1e-4

    # Create dataloaders
    train_loader, test_loader = create_sota_dataloaders(
        batch_size=batch_size,
        use_precomputed=True,
        num_workers=0 if torch.cuda.is_available() else 0
    )

    # Initialize model
    model = FolkMusicCQTResNet(num_classes=len(GENRES), dropout_prob=0.2).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler_kwargs = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 5,
    }
    if 'verbose' in inspect.signature(optim.lr_scheduler.ReduceLROnPlateau).parameters:
        scheduler_kwargs['verbose'] = True

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    # SpecAugment for training augmentation
    spec_augment = SpecAugment(
        freq_mask_max=10,
        time_mask_max=30,
        num_freq_masks=2,
        num_time_masks=2
    )

    # Training loop
    best_test_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print(f'\nStarting training for {num_epochs} epochs...', flush=True)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}', flush=True)
        print('-' * 50, flush=True)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, spec_augment)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Evaluate
        test_loss, test_acc, report, cm = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Learning rate scheduling
        scheduler.step(test_loss)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'\n{report}')

        # Save best model
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            save_model(model, optimizer, epoch, test_loss, 'models/sota_best_model.pth')
            plot_confusion_matrix(cm, 'results/sota_confusion_matrix.png')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_model(model, optimizer, epoch, test_loss, f'models/sota_checkpoint_epoch_{epoch+1}.pth')

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')

    plt.tight_layout()
    plt.savefig('results/sota_training_curves.png')
    plt.close()

    print(f'\nTraining completed!')
    print(f'Best test accuracy: {best_test_accuracy:.2f}%')
    print(f'Training curves saved to results/sota_training_curves.png')

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()