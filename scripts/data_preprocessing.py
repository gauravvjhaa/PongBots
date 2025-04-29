import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)

def preprocess_tabular_data(df):
    """Preprocess tabular data for feed-forward networks"""
    print("Preprocessing tabular data...")
    
    # Feature engineering
    df['is_ball_moving_right'] = (df['ball_speed_x'] > 0).astype(int)
    df['is_ball_moving_down'] = (df['ball_speed_y'] > 0).astype(int)
    df['distance_to_left_paddle'] = df['ball_x']
    df['distance_to_right_paddle'] = 800 - df['ball_x']  # 800 is the screen width
    df['ball_speed'] = np.sqrt(df['ball_speed_x']**2 + df['ball_speed_y']**2)
    
    # Define features and target
    features = [
        'ball_x', 'ball_y', 'ball_speed_x', 'ball_speed_y',
        'is_ball_moving_right', 'is_ball_moving_down',
        'distance_to_left_paddle', 'distance_to_right_paddle',
        'ball_speed'
    ]
    
    X = df[features]
    y = df['optimal_left_paddle_position']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Tabular data preprocessed: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

def preprocess_image_data(X_images, y_images):
    """Preprocess image data for CNN"""
    print("Preprocessing image data...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_images, y_images, test_size=0.2, random_state=42)
    print(f"Image data preprocessed: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    return X_train, X_test, y_train, y_test

def preprocess_sequence_data(X_sequences, y_sequences):
    """Preprocess sequence data for LSTM"""
    print("Preprocessing sequence data...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)
    print(f"Sequence data preprocessed: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/tabular_data.csv')
    
    with open('data/image_data.pkl', 'rb') as f:
        X_images, y_images = pickle.load(f)
    
    with open('data/sequence_data.pkl', 'rb') as f:
        X_sequences, y_sequences = pickle.load(f)
    
    # Preprocess tabular data
    X_train_tab, X_test_tab, y_train_tab, y_test_tab, scaler_tab, features = preprocess_tabular_data(df)
    
    # Preprocess image data
    X_train_img, X_test_img, y_train_img, y_test_img = preprocess_image_data(X_images, y_images)
    
    # Preprocess sequence data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = preprocess_sequence_data(X_sequences, y_sequences)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    
    # Save tabular data
    with open('data/processed/tabular_data.pkl', 'wb') as f:
        pickle.dump((X_train_tab, X_test_tab, y_train_tab, y_test_tab), f)
    
    # Save scaler and features
    with open('data/processed/tabular_metadata.pkl', 'wb') as f:
        pickle.dump((scaler_tab, features), f)
    
    # Save image data
    with open('data/processed/image_data.pkl', 'wb') as f:
        pickle.dump((X_train_img, X_test_img, y_train_img, y_test_img), f)
    
    # Save sequence data
    with open('data/processed/sequence_data.pkl', 'wb') as f:
        pickle.dump((X_train_seq, X_test_seq, y_train_seq, y_test_seq), f)
    
    print("Preprocessing complete!")
