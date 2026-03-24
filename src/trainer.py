import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping
from model.proposed import build_srars_model

def save_as_parquet(df, filename):
    """Saves the given dataframe as a Parquet file for reproducibility."""
    save_path = os.path.join('data/processed', filename)
    df.to_parquet(save_path, compression='snappy')
    print(f"[INFO] Saved: {save_path}")

def prepare_numpy_arrays(df):
    """Converts embedding lists in the dataframe into numpy arrays for model input."""
    user_np = np.stack(df['user_vector'].values)
    item_np = np.stack(df['item_vector'].values)
    y_np = df['rating'].values
    return user_np, item_np, y_np

def run_training_pipeline(df, config): # config 인자 추가
    """
    Executes the full training and evaluation workflow:
    1. 7:1:2 Data Splitting
    2. Parquet Saving
    3. Model Training with EarlyStopping
    4. Metric Evaluation
    """
    # 1. Data Splitting (7:1:2 Ratio)
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)
    
    print(f"[INFO] Data split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Save Split Datasets
    os.makedirs('data/processed', exist_ok=True)
    save_as_parquet(train_df, 'train_data.parquet')
    save_as_parquet(val_df, 'val_data.parquet')
    save_as_parquet(test_df, 'test_data.parquet')

    # 3. Prepare Inputs
    x_train_user, x_train_item, y_train = prepare_numpy_arrays(train_df)
    x_val_user, x_val_item, y_val = prepare_numpy_arrays(val_df)
    x_test_user, x_test_item, y_test = prepare_numpy_arrays(test_df)

    # 4. Build and Train Model
    print("[INFO] Building and compiling the SRARS model...")
    # config를 build_srars_model에 전달합니다.
    model = build_srars_model(config) 
    
    # config에서 learning_rate 등을 가져와 설정할 수 있습니다.
    lr = config['train'].get('learning_rate', 0.001)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )

    # config에서 patience 등을 가져옵니다.
    patience = config['train'].get('patience', 5)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=patience,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    batch_size = config['train'].get('batch_size', 64)
    epochs = config['train'].get('epochs', 50)

    print("\n--- Training Process Started ---")
    model.fit(
        x=[x_train_user, x_train_item],
        y=y_train,
        validation_data=([x_val_user, x_val_item], y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # 5. Final Evaluation
    print("\n--- Final Performance Metrics ---")
    predictions = model.predict([x_test_user, x_test_item])
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    print(f"Final MAE  : {mae:.4f}")
    print(f"Final RMSE : {rmse:.4f}")
    print(f"Final MAPE : {mape:.4f}%")

    # 6. Save Model
    os.makedirs('model', exist_ok=True)
    save_path = config['paths'].get('model_save', 'model/srars_best_model.h5')
    model.save(save_path)
    print(f"\n[SUCCESS] Best model weights saved to '{save_path}'")