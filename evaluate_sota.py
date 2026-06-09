# new/evaluate_sota.py — Comprehensive SOTA Model Evaluation
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Anchor CWD to the `new/` directory so relative paths like `models/`
# and `results/` always resolve inside `new/`.
NEW_ROOT = Path(__file__).resolve().parent
os.chdir(str(NEW_ROOT))

sys.path.append(str(NEW_ROOT))
from model import FolkMusicCQTResNet
from dataset import create_sota_dataloaders
from utils import GENRES, print_settings

def load_model(model_path, device):
    """Load trained model from checkpoint"""
    model = FolkMusicCQTResNet(num_classes=len(GENRES), dropout_prob=0.2).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'Model loaded from {model_path}')
    print(f'Epoch: {checkpoint.get("epoch", "N/A")}, Loss: {checkpoint.get("loss", "N/A")}')
    
    return model

def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Classification report
    class_report = classification_report(all_labels, all_preds, target_names=GENRES, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Confidence analysis
    confidence_scores = np.max(all_probs, axis=1)
    avg_confidence = np.mean(confidence_scores)
    
    # Analysis of correct vs incorrect predictions
    correct_mask = all_preds == all_labels
    correct_confidence = np.mean(confidence_scores[correct_mask])
    incorrect_confidence = np.mean(confidence_scores[~correct_mask])
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'avg_confidence': avg_confidence,
        'correct_confidence': correct_confidence,
        'incorrect_confidence': incorrect_confidence,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }

def plot_confusion_matrix(cm, save_path, title='Confusion Matrix'):
    """Plot confusion matrix with improved styling"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=GENRES, yticklabels=GENRES,
                cbar_kws={'label': 'Number of Samples'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(metrics, save_path):
    """Plot per-class precision, recall, and F1 scores"""
    x = np.arange(len(GENRES))
    width = 0.25
    
    plt.figure(figsize=(15, 8))
    plt.bar(x - width, metrics['precision_per_class'], width, label='Precision', alpha=0.8)
    plt.bar(x, metrics['recall_per_class'], width, label='Recall', alpha=0.8)
    plt.bar(x + width, metrics['f1_per_class'], width, label='F1 Score', alpha=0.8)
    
    plt.xlabel('Genres')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, GENRES, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(metrics, save_path):
    """Plot confidence distribution for correct vs incorrect predictions"""
    all_labels = metrics['all_labels']
    all_preds = metrics['all_preds']
    all_probs = metrics['all_probs']
    confidence_scores = np.max(all_probs, axis=1)
    
    correct_mask = all_preds == all_labels
    correct_confidence = confidence_scores[correct_mask]
    incorrect_confidence = confidence_scores[~correct_mask]
    
    plt.figure(figsize=(12, 6))
    plt.hist(correct_confidence, bins=30, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(incorrect_confidence, bins=30, alpha=0.7, label='Incorrect Predictions', color='red')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_evaluation_report(metrics, save_path):
    """Generate comprehensive evaluation report"""
    report = f"""
# SOTA Model Evaluation Report

## Overall Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Macro Precision**: {metrics['precision_macro']:.4f}
- **Macro Recall**: {metrics['recall_macro']:.4f}
- **Macro F1 Score**: {metrics['f1_macro']:.4f}
- **Weighted Precision**: {metrics['precision_weighted']:.4f}
- **Weighted Recall**: {metrics['recall_weighted']:.4f}
- **Weighted F1 Score**: {metrics['f1_weighted']:.4f}

## Confidence Analysis
- **Average Confidence**: {metrics['avg_confidence']:.4f}
- **Correct Prediction Confidence**: {metrics['correct_confidence']:.4f}
- **Incorrect Prediction Confidence**: {metrics['incorrect_confidence']:.4f}

## Per-Class Performance
"""
    
    for i, genre in enumerate(GENRES):
        report += f"""
### {genre}
- **Precision**: {metrics['precision_per_class'][i]:.4f}
- **Recall**: {metrics['recall_per_class'][i]:.4f}
- **F1 Score**: {metrics['f1_per_class'][i]:.4f}
"""
    
    # Add detailed classification report
    report += """
## Detailed Classification Report
"""
    for genre in GENRES:
        report += f"""
### {genre}
- Precision: {metrics['classification_report'][genre]['precision']:.4f}
- Recall: {metrics['classification_report'][genre]['recall']:.4f}
- F1-Score: {metrics['classification_report'][genre]['f1-score']:.4f}
- Support: {metrics['classification_report'][genre]['support']}
"""
    
    report += f"""
## Macro Average
- Precision: {metrics['classification_report']['macro avg']['precision']:.4f}
- Recall: {metrics['classification_report']['macro avg']['recall']:.4f}
- F1-Score: {metrics['classification_report']['macro avg']['f1-score']:.4f}

## Weighted Average
- Precision: {metrics['classification_report']['weighted avg']['precision']:.4f}
- Recall: {metrics['classification_report']['weighted avg']['recall']:.4f}
- F1-Score: {metrics['classification_report']['weighted avg']['f1-score']:.4f}
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f'Detailed evaluation report saved to {save_path}')

def main():
    # Initialize settings
    print_settings()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Paths
    model_path = 'models/sota_best_model.pth'
    results_dir = 'results'
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model
    if not os.path.exists(model_path):
        print(f'Model not found at {model_path}')
        print('Please train the model first using: python new/train_sota.py')
        return
    
    model = load_model(model_path, device)
    
    # Create dataloaders
    train_loader, test_loader = create_sota_dataloaders(
        batch_size=16, 
        use_precomputed=True,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    # Evaluate model
    print('\nStarting comprehensive evaluation...')
    metrics = evaluate_model(model, test_loader, device)
    
    # Generate visualizations
    print('\nGenerating visualizations...')
    plot_confusion_matrix(metrics['confusion_matrix'], f'{results_dir}/sota_confusion_matrix.png')
    plot_per_class_metrics(metrics, f'{results_dir}/sota_per_class_metrics.png')
    plot_confidence_distribution(metrics, f'{results_dir}/sota_confidence_distribution.png')
    
    # Generate report
    generate_evaluation_report(metrics, f'{results_dir}/sota_evaluation_report.md')
    
    # Print summary
    print('\n=== EVALUATION SUMMARY ===')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Macro F1 Score: {metrics["f1_macro"]:.4f}')
    print(f'Average Confidence: {metrics["avg_confidence"]:.4f}')
    print(f'Correct vs Incorrect Confidence: {metrics["correct_confidence"]:.4f} vs {metrics["incorrect_confidence"]:.4f}')
    print('\nVisualizations and report saved to results/ folder')

if __name__ == "__main__":
    main()