# new/predict_sota.py — SOTA Audio Classification Prediction Script
import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import librosa
from tqdm import tqdm

# Anchor CWD to the `new/` directory so relative paths like `models/`
# always resolve inside `new/`.
NEW_ROOT = Path(__file__).resolve().parent
os.chdir(str(NEW_ROOT))

sys.path.append(str(NEW_ROOT))
from model import FolkMusicCQTResNet
from preprocess import compute_sota_features
from utils import GENRES, SAMPLE_RATE, DURATION, print_settings

class SOTAPredictor:
    """
    SOTA Audio Classifier using CQT + HPSS features and ResNet model
    """
    def __init__(self, model_path, device='auto'):
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.genre_names = GENRES
        print(f"Model loaded: {model_path}")
        print(f"Device: {self.device}")
    
    def _get_device(self, device):
        """Determine the best available device"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self, model_path):
        """Load the trained SOTA model"""
        model = FolkMusicCQTResNet(num_classes=len(self.genre_names), dropout_prob=0.2)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model.to(self.device)
    
    def predict_audio_file(self, audio_path, chunk_overlap=0.5, show_chunks=False):
        """
        Predict genre of an audio file by processing it in chunks and aggregating results
        
        Args:
            audio_path: Path to audio file
            chunk_overlap: Overlap between chunks (0-1)
            show_chunks: Whether to show individual chunk predictions
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load audio file
            audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            
            # Split into chunks
            chunks = self._split_audio_into_chunks(audio, sr, chunk_overlap)
            
            if len(chunks) == 0:
                raise ValueError(f"Could not split audio into chunks: {audio_path}")
            
            # Process each chunk
            chunk_predictions = []
            chunk_confidences = []
            
            for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                # Extract SOTA features
                features = compute_sota_features(chunk, sr)
                
                # Convert to tensor and add batch dimension
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, dim=1)
                
                chunk_predictions.append(predicted.item())
                chunk_confidences.append(confidence.item())
                
                if show_chunks:
                    print(f"Chunk {i+1}: {self.genre_names[predicted.item()]} ({confidence.item():.3f})")
            
            # Aggregate predictions
            avg_probs = np.zeros(len(self.genre_names))
            for genre_idx in range(len(self.genre_names)):
                avg_probs[genre_idx] = np.mean([chunk_confidences[i] for i, pred in enumerate(chunk_predictions) if pred == genre_idx])
            
            # Get final prediction
            final_prediction_idx = np.argmax(avg_probs)
            final_prediction = self.genre_names[final_prediction_idx]
            final_confidence = avg_probs[final_prediction_idx]
            
            # Calculate prediction statistics
            consistency_ratio = self._calculate_consistency(chunk_predictions, final_prediction_idx)
            entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-12))
            
            # Determine if prediction is reliable
            is_reliable = self._is_prediction_reliable(
                final_confidence, entropy, consistency_ratio, 
                min_confidence=0.6, min_consistency=0.6, max_entropy=1.0
            )
            
            results = {
                'file_path': str(audio_path),
                'predicted_genre': final_prediction if is_reliable else None,
                'confidence': float(final_confidence),
                'entropy': float(entropy),
                'consistency_ratio': float(consistency_ratio),
                'is_reliable': is_reliable,
                'chunk_predictions': [self.genre_names[p] for p in chunk_predictions],
                'chunk_confidences': chunk_confidences,
                'average_probabilities': {genre: float(prob) for genre, prob in zip(self.genre_names, avg_probs)},
                'num_chunks': len(chunks),
                'rejection_reason': None if is_reliable else self._get_rejection_reason(final_confidence, entropy, consistency_ratio)
            }
            
            return results
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return {
                'file_path': str(audio_path),
                'error': str(e),
                'predicted_genre': None,
                'confidence': 0.0,
                'is_reliable': False
            }
    
    def _split_audio_into_chunks(self, audio, sr, overlap=0.5):
        """Split audio into overlapping chunks"""
        chunk_length = int(DURATION * sr)
        step_size = int(chunk_length * (1 - overlap))
        
        chunks = []
        for start in range(0, len(audio), step_size):
            end = start + chunk_length
            chunk = audio[start:end]
            
            # Pad if necessary
            if len(chunk) < chunk_length:
                padding = np.zeros(chunk_length - len(chunk))
                chunk = np.concatenate([chunk, padding])
            
            chunks.append(chunk)
        
        return chunks
    
    def _calculate_consistency(self, predictions, final_prediction_idx):
        """Calculate how many chunks agree with the final prediction"""
        agreement_count = sum(1 for pred in predictions if pred == final_prediction_idx)
        return agreement_count / len(predictions) if len(predictions) > 0 else 0
    
    def _is_prediction_reliable(self, confidence, entropy, consistency_ratio, 
                               min_confidence=0.6, min_consistency=0.6, max_entropy=1.0):
        """Check if prediction meets reliability criteria"""
        return (
            confidence >= min_confidence and
            consistency_ratio >= min_consistency and
            entropy <= max_entropy
        )
    
    def _get_rejection_reason(self, confidence, entropy, consistency_ratio):
        """Get reason why prediction was rejected"""
        reasons = []
        if confidence < 0.6:
            reasons.append(f"Low confidence ({confidence:.3f})")
        if entropy > 1.0:
            reasons.append(f"High entropy ({entropy:.3f})")
        if consistency_ratio < 0.6:
            reasons.append(f"Low consistency ({consistency_ratio:.3f})")
        return "; ".join(reasons) if reasons else "Unknown reason"
    
    def predict_random_file_from_dir(self, directory, recursive=True):
        """Pick a random audio file from directory and classify it"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.webm', '.m4a', '.mp4', '.ogg', '.opus'}
        
        audio_files = []
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if Path(file).suffix.lower() in audio_extensions:
                        audio_files.append(Path(root) / file)
        else:
            for file in os.listdir(directory):
                if Path(file).suffix.lower() in audio_extensions:
                    audio_files.append(Path(directory) / file)
        
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {directory}")
        
        random_file = random.choice(audio_files)
        return self.predict_audio_file(random_file, show_chunks=True)
    
    def batch_predict(self, audio_files_or_dirs, recursive=True, output_file=None):
        """Predict multiple files or directories"""
        results = []
        
        for path in tqdm(audio_files_or_dirs, desc="Processing files"):
            path_obj = Path(path)
            
            if path_obj.is_file():
                result = self.predict_audio_file(path_obj)
                results.append(result)
            elif path_obj.is_dir():
                try:
                    dir_result = self.predict_random_file_from_dir(path_obj, recursive)
                    results.append(dir_result)
                except Exception as e:
                    results.append({
                        'file_path': str(path),
                        'error': str(e),
                        'predicted_genre': None,
                        'is_reliable': False
                    })
        
        # Save results if output file specified
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="SOTA Nepali Folk Music Classification")
    parser.add_argument("--file", type=str, help="Path to audio file to classify")
    parser.add_argument("--dir", type=str, help="Directory containing audio files")
    parser.add_argument("--recursive", action="store_true", default=True, 
                       help="Search directories recursively (default: True)")
    parser.add_argument("--model", type=str, default="models/sota_best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--show-chunks", action="store_true",
                       help="Show individual chunk predictions")
    parser.add_argument("--random", action="store_true",
                       help="Pick a random file from directory")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Initialize settings
    print_settings()
    
    # Initialize predictor
    predictor = SOTAPredictor(args.model, args.device)
    
    # Determine what to predict
    if args.file:
        # Single file prediction
        result = predictor.predict_audio_file(args.file, show_chunks=args.show_chunks)
        print(f"\n=== PREDICTION RESULTS ===")
        print(f"File: {result['file_path']}")
        print(f"Chunks processed: {result['num_chunks']}")
        
        if result['predicted_genre']:
            print(f"Predicted Genre: {result['predicted_genre']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Consistency: {result['consistency_ratio']:.3f}")
            print(f"Entropy: {result['entropy']:.3f}")
        else:
            print("Genre not confidently identified")
            if result.get('rejection_reason'):
                print(f"Reason: {result['rejection_reason']}")
        
        # Show top probabilities
        print("\nAll Probabilities:")
        for genre, prob in sorted(result['average_probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {genre}: {prob:.3f}")
    
    elif args.dir:
        # Directory prediction
        if args.random:
            result = predictor.predict_random_file_from_dir(args.dir, args.recursive)
            print(f"\n=== RANDOM PREDICTION FROM {args.dir} ===")
            print(f"File: {result['file_path']}")
            print(f"Predicted Genre: {result['predicted_genre']}")
            print(f"Confidence: {result['confidence']:.3f}")
        else:
            audio_files = []
            for root, dirs, files in os.walk(args.dir):
                for file in files:
                    if Path(file).suffix.lower() in {'.wav', '.mp3', '.flac', '.webm', '.m4a', '.mp4', '.ogg', '.opus'}:
                        audio_files.append(Path(root) / file)
            
            if not audio_files:
                print(f"No audio files found in {args.dir}")
                return
            
            results = predictor.batch_predict(audio_files, args.recursive, args.output)
            
            # Print summary
            successful = [r for r in results if r['predicted_genre']]
            print(f"\n=== SUMMARY ===")
            print(f"Total files processed: {len(results)}")
            print(f"Successful predictions: {len(successful)}")
            print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
            
            # Genre distribution
            genre_counts = {}
            for result in successful:
                genre = result['predicted_genre']
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            print("\nGenre Distribution:")
            for genre, count in sorted(genre_counts.items()):
                print(f"  {genre}: {count}")
    
    else:
        print("Please specify --file or --dir argument")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()