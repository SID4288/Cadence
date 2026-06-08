import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add parent directory
from utils import GENRES
import joblib


def load_split_features(data_dir, split_name):
    split_dir = os.path.join(data_dir, split_name)
    X = np.load(os.path.join(split_dir, 'X_features.npy'))
    Y = np.load(os.path.join(split_dir, 'y_labels.npy'))
    file_paths_path = os.path.join(split_dir, 'file_paths.npy')
    file_paths = np.load(file_paths_path, allow_pickle=True) if os.path.exists(file_paths_path) else None
    return X, Y, file_paths

def generate_song_groups_from_file_paths(file_paths):
    """
    Maps every chunk belonging to the same source song to a unique integer group ID.
    """
    groups = []
    song_to_id_map = {}
    current_id = 0

    for filepath in file_paths:
        filename = os.path.basename(str(filepath))
        # Extract base song token (e.g., 'tamang_selo_songA_chunk1.wav' -> 'tamang_selo_songA')
        parts = filename.split("_")
        song_id_string = "_".join(parts[:-1])

        if song_id_string not in song_to_id_map:
            song_to_id_map[song_id_string] = current_id
            current_id += 1
            
        groups.append(song_to_id_map[song_id_string])

    return np.array(groups)


def generate_song_groups_from_split_root(split_root_dir):
    """
    Fallback grouping helper that reconstructs the same walk order used by
    src/features.py when file_paths.npy is unavailable.
    """
    file_paths = []

    for genre in GENRES:
        genre_dir = os.path.join(split_root_dir, genre)
        if not os.path.exists(genre_dir):
            continue

        clips = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        for clip_name in clips:
            file_paths.append(os.path.join(genre_dir, clip_name))

    return generate_song_groups_from_file_paths(file_paths)

def prepare_data(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
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
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GENRES, yticklabels=GENRES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    accuracy = accuracy_score(y_test, y_pred)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
    plt.savefig('results/confusion_matrix.png')
    plt.close() # Prevent window lock during background runs
    return model

def cross_validate(X_train_raw, y_train, groups_train):
    print("\nRunning 5-fold GROUP cross validation on the train split...")

    # C matches your cross-validation setup (C=10)
    svm_cv = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))
    ])
    
    # Strictly isolate chunks belonging to the same song id using GroupKFold
    kfold = GroupKFold(n_splits=5)
    
    scores = cross_val_score(
        svm_cv, 
        X_train_raw, 
        y_train, 
        groups=groups_train, 
        cv=kfold, 
        scoring='f1_macro'
    )
    
    print(f"F1 scores per fold: {scores.round(3)}")
    print(f"Mean F1: {scores.mean():.3f} ± {scores.std():.3f}")

def save_model(svm, scaler):
    os.makedirs("models/classifier", exist_ok=True)
    joblib.dump(svm, "models/classifier/svm_model.pkl")
    joblib.dump(scaler, "models/classifier/scaler.pkl")
    print("Model saved to models/classifier/")


if __name__ == "__main__":
    # 1. Load your processed matrices
    X_train_raw, y_train, train_files = load_split_features("data/features", "train")
    X_test_raw, y_test, test_files = load_split_features("data/features", "test")
    
    # 2. Extract song groupings from the saved file paths to safely prevent data leakage
    print("Extracting song identities from saved feature metadata to prevent data leakage...")
    if train_files is not None:
        groups_train = generate_song_groups_from_file_paths(train_files)
    else:
        print("Saved file_paths.npy not found; falling back to data/splits/train folder walk.")
        groups_train = generate_song_groups_from_split_root("data/splits/train")
    
    # 3. Assert shapes to ensure features map 1:1 with file records
    assert len(X_train_raw) == len(groups_train), \
        f"Mismatch error: You have {len(X_train_raw)} entries in feature matrix but {len(groups_train)} files in the train split."

    # 4. Standard Scaler Transform
    X_train, X_test, y_train, y_test, scaler = prepare_data(X_train_raw, y_train, X_test_raw, y_test)
    
    # 5. Evaluate and cross-validate safely
    model = train_and_evaluate(X_train, y_train, X_test, y_test)
    cross_validate(X_train_raw, y_train, groups_train)
    
    # 6. Save model configuration
    save_model(model, scaler)
