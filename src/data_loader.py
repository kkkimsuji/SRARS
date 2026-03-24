import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import gzip
import json

def load_and_preprocess(file_path):
    """
    Load raw Amazon review data and perform basic cleaning and filtering.
    Based on the paper: 'Utilizing Advanced Transformers for Summarized Review-Aware Recommender System'
    """
    # 1. Load Dataset (Using lines=True for Amazon JSONL format)
    # This is more efficient than manual line-by-line parsing in VSCode environment.
    df = pd.read_json(file_path, compression='gzip', lines=True)
    
    # 2. Select essential columns and rename them for consistency [cite: 233]
    # user: reviewerID, item: asin, review: reviewText, rating: overall
    df = df[['reviewerID', 'asin', 'reviewText', 'overall']]
    df.columns = ['user', 'item', 'review', 'rating']
    
    # 3. Basic Data Cleaning
    df = df.drop_duplicates()
    df['review'] = df['review'].fillna('') # Handle missing review texts [cite: 172]
    
    # 4. 5-core Filtering [cite: 335]
    # Filter users who have written at least 5 reviews to ensure sufficient preference data.
    user_counts = df['user'].value_counts()
    df = df[df['user'].isin(user_counts[user_counts >= 5].index)]
    
    # 5. Label Encoding [cite: 369]
    # Convert string IDs (reviewerID, asin) into numerical indices for model embedding.
    le_user, le_item = LabelEncoder(), LabelEncoder()
    df['user'] = le_user.fit_transform(df['user'])
    df['item'] = le_item.fit_transform(df['item'])
    
    return df

def generate_review_sets(df):
    """
    Aggregate individual reviews into User Review Sets (D^u) and Item Review Sets (D^v).
    These sets are used as input for the BART summarization module. [cite: 238, 239, 240]
    """
    # 1. Aggregate all reviews written by each user (User Preference Representation)
    user_reviews = df.groupby('user')['review'].apply(lambda x: " ".join(x)).reset_index()
    user_reviews.columns = ['user', 'user_review_set']
    
    # 2. Aggregate all reviews written for each item (Item Attribute Representation)
    item_reviews = df.groupby('item')['review'].apply(lambda x: " ".join(x)).reset_index()
    item_reviews.columns = ['item', 'item_review_set']
    
    # 3. Merge aggregated review sets back into the original dataframe [cite: 231]
    df = pd.merge(df, user_reviews, on='user', how='left')
    df = pd.merge(df, item_reviews, on='item', how='left')
    
    return df

if __name__ == "__main__":
    # Define file paths for the local VSCode environment
    # Ensure SampleData.json.gz is placed in 'data/raw/' directory
    input_path = 'data/raw/SampleData.json.gz'
    output_dir = 'data/processed'
    
    print("Step 1: Loading and Filtering Data...")
    processed_df = load_and_preprocess(input_path)
    
    print("Step 2: Generating User/Item Review Sets...")
    final_df = generate_review_sets(processed_df)
    
    # Save the preprocessed dataframe for the next stage (BART Embedding)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Using pickle format to preserve dataframe structure and types
    final_df.to_pickle(os.path.join(output_dir, 'preprocessed_data.pkl'))
    
    print(f"Successfully processed {len(final_df)} interactions.")
    print(f"Data saved to {output_dir}/preprocessed_data.pkl")