import os
import yaml
import pandas as pd
# Import modular functions
from src.data_loader import load_and_preprocess, generate_review_sets
from src.bart import bart_embedding
from src.trainer import run_training_pipeline

def main():
    # 0. Load Configuration
    config_path = 'src/config.yaml'
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("SRARS: Summarized Review-Aware Recommender System")
    print("Full Pipeline: Preprocessing -> Embedding -> Training")
    print("="*60)

    # Step 1: Data Preprocessing
    # Use paths defined in config.yaml
    processed_dir = config['paths']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    processed_path = os.path.join(processed_dir, 'preprocessed_data.pkl')
    
    if not os.path.exists(processed_path):
        print("[STEP 1] Preprocessing raw data...")
        raw_data_path = config['paths']['raw_data']
        raw_df = load_and_preprocess(raw_data_path)
        df = generate_review_sets(raw_df)
        df.to_pickle(processed_path)
    else:
        print("[STEP 1] Found existing preprocessed data. Skipping...")
        df = pd.read_pickle(processed_path)

    # Step 2: Feature Extraction (BART)
    embedded_path = os.path.join(processed_dir, 'embedded_data.pkl')
    
    if not os.path.exists(embedded_path):
        print("[STEP 2] Extracting BART embeddings (This may take some time)...")
        # You can also pass model_name from config if added to yaml
        user_vecs = bart_embedding(df, 'user_review_set')
        item_vecs = bart_embedding(df, 'item_review_set')
        
        df['user_vector'] = list(user_vecs)
        df['item_vector'] = list(item_vecs)
        df.to_pickle(embedded_path)
    else:
        print("[STEP 2] Found existing embeddings. Skipping...")
        df = pd.read_pickle(embedded_path)

    # Step 3: Training & Evaluation
    print("[STEP 3] Entering Training Pipeline...")
    # Pass the config dictionary to the trainer
    run_training_pipeline(df, config)

    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()