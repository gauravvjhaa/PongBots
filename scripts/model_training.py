import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import pickle
import time
from datetime import datetime

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Custom callback to track hit rate
class HitRateMonitor(Callback):
    def __init__(self, validation_data, hit_margin=10.0):
        super(HitRateMonitor, self).__init__()
        self.validation_data = validation_data
        self.hit_margin = hit_margin
        self.best_hit_rate = -1
    
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val, verbose=0)
        
        # Calculate hit rate based on predictions within margin
        hits = np.sum(np.abs(y_pred.flatten() - y_val) <= self.hit_margin)
        hit_rate = hits / len(y_val)
        
        logs = logs or {}
        logs['hit_rate'] = hit_rate
        
        print(f' - hit_rate: {hit_rate:.4f}')
        
        # Track if this is best hit rate
        if hit_rate > self.best_hit_rate:
            self.best_hit_rate = hit_rate
            print(f'New best hit rate: {hit_rate:.4f}')
            return True
        return False

# Custom metric for hit rate
def hit_rate_metric(hit_margin=10.0):
    def hit_rate(y_true, y_pred):
        # Count predictions within margin as hits
        hits = tf.reduce_sum(tf.cast(tf.abs(y_pred - y_true) <= hit_margin, tf.float32))
        return hits / tf.cast(tf.shape(y_true)[0], tf.float32)
    return hit_rate

def build_feed_forward_model(input_shape):
    """Build a simple feed-forward neural network"""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_shape),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='mse',
        metrics=[hit_rate_metric(10.0)]
    )
    return model

def build_deep_feed_forward_model(input_shape):
    """Build a deeper feed-forward neural network with more layers"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005), 
        loss='mse', 
        metrics=[hit_rate_metric(10.0)]
    )
    return model

def build_cnn_model(input_shape):
    """Build a CNN model for image-based inputs"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=[hit_rate_metric(10.0)]
    )
    return model

def build_lstm_model(input_shape):
    """Build an LSTM model for sequence data"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=[hit_rate_metric(10.0)]
    )
    return model

def train_model(model, model_name, X_train, y_train, X_val, y_val, epochs=100):
    """Train a single model with hit rate monitoring"""
    print(f"\nTraining {model_name} model...")
    
    # Model save path
    model_path = f'models/{model_name.lower()}_model.keras'
    best_hit_rate = -1
    
    # Start timing
    start_time = time.time()
    
    # Prepare validation data tuple for hit rate monitor
    validation_data = (X_val, y_val)
    hit_monitor = HitRateMonitor(validation_data)
    
    # Train the model
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=1,
            verbose=1,
            validation_data=(X_val, y_val)
        )
        
        # Evaluate hit rate on validation data
        y_pred = model.predict(X_val, verbose=0)
        hits = np.sum(np.abs(y_pred.flatten() - y_val) <= 10.0)
        hit_rate = hits / len(y_val)
        
        print(f"Epoch {epoch+1}: hit_rate = {hit_rate:.4f}")
        
        # Save model if hit rate improved
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            print(f"New best hit rate: {hit_rate:.4f}, saving model to {model_path}")
            model.save(model_path)
    
    training_time = time.time() - start_time
    
    print(f"{model_name} model training complete in {training_time:.2f} seconds")
    print(f"Best hit rate achieved: {best_hit_rate:.4f}")
    
    return model, training_time, best_hit_rate

def train_all_models():
    """Train all models one by one, monitoring hit rate"""
    # Load preprocessed data
    print("Loading preprocessed data...")
    
    # Load tabular data
    with open('data/processed/tabular_data.pkl', 'rb') as f:
        X_train_tab, X_test_tab, y_train_tab, y_test_tab = pickle.load(f)
    
    # Load image data
    with open('data/processed/image_data.pkl', 'rb') as f:
        X_train_img, X_test_img, y_train_img, y_test_img = pickle.load(f)
    
    # Load sequence data
    with open('data/processed/sequence_data.pkl', 'rb') as f:
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = pickle.load(f)
    
    # Split training data into train and validation
    val_split = 0.2
    val_idx_tab = int(X_train_tab.shape[0] * (1 - val_split))
    val_idx_img = int(X_train_img.shape[0] * (1 - val_split))
    val_idx_seq = int(X_train_seq.shape[0] * (1 - val_split))
    
    X_val_tab = X_train_tab[val_idx_tab:]
    y_val_tab = y_train_tab[val_idx_tab:]
    X_train_tab = X_train_tab[:val_idx_tab]
    y_train_tab = y_train_tab[:val_idx_tab]
    
    X_val_img = X_train_img[val_idx_img:]
    y_val_img = y_train_img[val_idx_img:]
    X_train_img = X_train_img[:val_idx_img]
    y_train_img = y_train_img[:val_idx_img]
    
    X_val_seq = X_train_seq[val_idx_seq:]
    y_val_seq = y_train_seq[val_idx_seq:]
    X_train_seq = X_train_seq[:val_idx_seq]
    y_train_seq = y_train_seq[:val_idx_seq]
    
    # Storage for results
    training_times = {}
    hit_rates = {}
    
    # 1. Train Simple Feed-Forward Neural Network
    simple_ff_model = build_feed_forward_model(X_train_tab.shape[1])
    _, training_time, best_hit_rate = train_model(
        simple_ff_model, 
        'SimpleFF', 
        X_train_tab, y_train_tab,
        X_val_tab, y_val_tab,
        epochs=100
    )
    training_times['SimpleFF'] = training_time
    hit_rates['SimpleFF'] = best_hit_rate
    
    # 2. Train Deep Feed-Forward Neural Network
    deep_ff_model = build_deep_feed_forward_model(X_train_tab.shape[1])
    _, training_time, best_hit_rate = train_model(
        deep_ff_model, 
        'DeepFF', 
        X_train_tab, y_train_tab,
        X_val_tab, y_val_tab,
        epochs=100
    )
    training_times['DeepFF'] = training_time
    hit_rates['DeepFF'] = best_hit_rate
    
    # 3. Train CNN Model
    cnn_model = build_cnn_model(X_train_img.shape[1:])
    _, training_time, best_hit_rate = train_model(
        cnn_model, 
        'CNN', 
        X_train_img, y_train_img,
        X_val_img, y_val_img,
        epochs=100
    )
    training_times['CNN'] = training_time
    hit_rates['CNN'] = best_hit_rate
    
    # 4. Train LSTM Model
    lstm_model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    _, training_time, best_hit_rate = train_model(
        lstm_model, 
        'LSTM', 
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=100
    )
    training_times['LSTM'] = training_time
    hit_rates['LSTM'] = best_hit_rate
    
    # Save training results
    results = {
        'training_times': training_times,
        'hit_rates': hit_rates,
        'timestamp': '2025-04-29 10:49:15'
    }
    
    with open('results/training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Date: 2025-04-29 10:49:15")
    print(f"User: gauravvjhaa")
    print("\nModel Performance:")
    for model_name in training_times.keys():
        print(f"  {model_name}:")
        print(f"    - Training time: {training_times[model_name]:.2f} seconds")
        print(f"    - Best hit rate: {hit_rates[model_name]:.4f}")
    
    return results

if __name__ == "__main__":
    train_all_models()
