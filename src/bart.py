import pandas as pd
import numpy as np
import torch
from transformers import BartTokenizer, BartModel
from tqdm import tqdm
import os

def bart_embedding(df, text_col, batch_size=1, max_length=768):
    """
    Extracts review embeddings using the BART encoder's last hidden state.
    
    Strictly follows the paper's original implementation: 
    'Utilizing Advanced Transformers for Summarized Review-Aware Recommender System'.
    The hidden state of the last token is selected using [:, -1, :][0].
    """
    model_name = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Device configuration (Automatic detection)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BartModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    text_list = df[text_col].tolist()

    print(f"[INFO] Extracting BART embeddings for '{text_col}' using {device}...")
    
    # Loop with batch_size=1 to ensure precise vector extraction as per paper
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_reviews = text_list[i:i + batch_size]
        
        inputs = tokenizer(
            batch_reviews, 
            return_tensors='pt', 
            max_length=max_length, 
            truncation=True, 
            padding='max_length'
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # encoder_hidden_states: (batch_size, sequence_length, hidden_size)
            encoder_hidden_states = outputs.last_hidden_state

            # Precisely following the paper's logic: Select the last token and take the first item
            # Shape transition: (1, seq, 768) -> (1, 768) -> (768,)
            result = encoder_hidden_states[:, -1, :][0].cpu().numpy()
            embeddings.append(result)

    # Stack individual 1D vectors into a single 2D matrix (N, 768)
    stacked_embeddings = np.vstack(embeddings)
    return stacked_embeddings

if __name__ == "__main__":
    # Local path configuration
    input_path = 'data/processed/preprocessed_data.pkl'
    output_path = 'data/processed/embedded_data.pkl'

    if os.path.exists(input_path):
        df = pd.read_pickle(input_path)
        
        # User (U0) and Item (I0) feature extraction
        user_vecs = bart_embedding(df, 'user_review_set', batch_size=1)
        item_vecs = bart_embedding(df, 'item_review_set', batch_size=1)

        # Assign extracted vectors back to the dataframe
        df['user_vector'] = list(user_vecs)
        df['item_vector'] = list(item_vecs)

        # Ensure directory exists and save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_pickle(output_path)
        print(f"[SUCCESS] Embedded data saved to {output_path}")
    else:
        print(f"[ERROR] '{input_path}' not found. Please run data_loader.py first.")