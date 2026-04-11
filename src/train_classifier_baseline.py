import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add parent directory
from utils import GENRES
import joblib

def load_data(data_dir):
    X = np.load(os.path.join(data_dir, 'X_features.npy'))
    Y = np.load(os.path.join(data_dir, 'y_labels.npy'))
    return X, Y

def prepare_data(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42,
        stratify= Y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=GENRES))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GENRES, yticklabels=GENRES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    accuracy = accuracy_score(y_test, y_pred)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
    plt.savefig('results/confusion_matrix.png')
    plt.show()
    return model

def cross_validate(X, Y, scaler):
    print("\nRunning 5-fold cross validation...")
    
    X_scaled = scaler.transform(X)
    
    svm_cv = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(svm_cv, X_scaled, Y, cv=kfold, scoring='f1_macro')
    
    print(f"F1 scores per fold: {scores.round(3)}")
    print(f"Mean F1: {scores.mean():.3f} ± {scores.std():.3f}")

def save_model(svm, scaler):
    os.makedirs("models/classifier", exist_ok=True)
    joblib.dump(svm, "models/classifier/svm_model.pkl")
    joblib.dump(scaler, "models/classifier/scaler.pkl")
    print("Model saved to models/classifier/")

if __name__ == "__main__":
    X, Y = load_data("data/features")
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, Y)
    model = train_and_evaluate(X_train, y_train, X_test, y_test)
    cross_validate(X, Y, scaler)
    save_model(model, scaler)