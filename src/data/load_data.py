import pandas as pd
import os
from pathlib import Path

def load_raw_data():
    """
    Load Titanic dataset from data/raw/
    Works regardless of where the function is called from (notebook, script, etc.)
    """
    project_root = Path(__file__).parent.parent.parent
    
    train_path = project_root / "data" / "raw" / "train.csv"
    
    # Fallback si les fichiers n'existent pas encore 
    if not train_path.exists():
        print(f"Fichier non trouvé à {train_path}")
    
    train = pd.read_csv(train_path)
    
    return train

def save_processed_data(df, filename):
    """Save processed dataframe in data/processed/"""
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = processed_dir / filename
    df.to_csv(filepath, index=False)
    print(f" Saved: {filepath}")

# Test rapide si exécuté directement
if __name__ == "__main__":
    train = load_raw_data()
    print(f"Train shape: {train.shape}")