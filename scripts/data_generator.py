import os
import numpy as np
import pandas as pd
from pong_simulator import PongSimulator
import pickle

# Create directory for outputs
os.makedirs('data', exist_ok=True)

def generate_training_data(n_games=100, steps_per_game=1000, difficulty='medium'):
    """Generate training data from simulated games"""
    print(f"Generating training data: {n_games} games x {steps_per_game} steps per game")
    simulator = PongSimulator(difficulty=difficulty)
    data = []
    image_data = []
    sequence_data = []
    
    for i in range(n_games):
        if i % 10 == 0:
            print(f"  Game {i}/{n_games}")
        simulator.reset_game()
        game_sequence = []
        
        for _ in range(steps_per_game):
            state = simulator.get_state()
            image = simulator.get_image_state()
            
            # Get optimal position for left paddle
            optimal_left_position = simulator.get_optimal_paddle_position(is_left_paddle=True)
            
            # Record state and optimal position
            data_point = {
                'ball_x': state['ball_x'],
                'ball_y': state['ball_y'],
                'ball_speed_x': state['ball_speed_x'],
                'ball_speed_y': state['ball_speed_y'],
                'paddle_left_y': state['paddle_left_y'],
                'paddle_right_y': state['paddle_right_y'],
                'optimal_left_paddle_position': optimal_left_position
            }
            data.append(data_point)
            
            # Save image state with same optimal position for CNN
            image_data.append((image, optimal_left_position))
            
            # Record sequence data point
            game_sequence.append(data_point)
            if len(game_sequence) >= 10:  # Once we have enough history
                sequence_data.append((game_sequence.copy(), optimal_left_position))
                game_sequence.pop(0)
            
            # Define a simple AI for the right paddle
            def right_paddle_ai(simulator):
                # Simple AI: follow the ball
                return simulator.ball_y - simulator.paddle_height / 2
            
            # Update the game state
            simulator.update(left_paddle_ai=None, right_paddle_ai=right_paddle_ai)
    
    # Convert tabular data to DataFrame
    df = pd.DataFrame(data)
    
    # Process image data
    X_images = np.array([img for img, _ in image_data])
    y_images = np.array([pos for _, pos in image_data])
    
    # Process sequence data
    X_sequences = []
    y_sequences = []
    for seq, target in sequence_data:
        # Extract features from each step in sequence
        seq_features = []
        for step in seq:
            features = [
                step['ball_x'], step['ball_y'], 
                step['ball_speed_x'], step['ball_speed_y'],
                step['paddle_left_y'], step['paddle_right_y']
            ]
            seq_features.append(features)
        X_sequences.append(seq_features)
        y_sequences.append(target)
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print("Data generation complete!")
    print(f"Tabular data: {len(df)} samples")
    print(f"Image data: {X_images.shape} images with {y_images.shape} labels")
    print(f"Sequence data: {X_sequences.shape} sequences with {y_sequences.shape} labels")
    
    return df, X_images, y_images, X_sequences, y_sequences

if __name__ == "__main__":
    # Generate data
    df, X_images, y_images, X_sequences, y_sequences = generate_training_data(
        n_games=100, 
        steps_per_game=1000, 
        difficulty='medium'
    )
    
    # Save data to disk
    print("Saving data to disk...")
    df.to_csv('data/tabular_data.csv', index=False)
    
    with open('data/image_data.pkl', 'wb') as f:
        pickle.dump((X_images, y_images), f)
    
    with open('data/sequence_data.pkl', 'wb') as f:
        pickle.dump((X_sequences, y_sequences), f)
    
    print("Data saved successfully!")
