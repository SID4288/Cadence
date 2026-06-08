import os 
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from utils import GENRES

data = []
for label, genre in enumerate(GENRES):
    genre_dir = f"data/raw/{genre}"
    files = [f for f in os.listdir(genre_dir) if f.endswith('.mp3') or f.endswith('.webm')]

    for filename in files:
        data.append({
            "filepath": os.path.join(genre_dir, filename),
            "label": label,
            "genre": genre
        })

df = pd.DataFrame(data)
df.to_csv("data/dataset.csv", index=False)
print(f"Dataset created with {len(df)} samples across {len(GENRES)} genres.")   
print(df["genre"].value_counts())