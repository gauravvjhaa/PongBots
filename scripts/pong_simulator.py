import numpy as np
import matplotlib.pyplot as plt

class PongSimulator:
    def __init__(self, width=800, height=400, ball_radius=10, paddle_width=10, paddle_height=60, 
                 paddle_speed=10, ball_speed_x=5, ball_speed_y=5, difficulty='medium'):
        self.width = width
        self.height = height
        self.ball_radius = ball_radius
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.paddle_speed = paddle_speed
        self.initial_ball_speed_x = ball_speed_x
        self.initial_ball_speed_y = ball_speed_y
        
        # Set difficulty-based parameters
        self.set_difficulty(difficulty)
        
        self.reset_game()
        
        # For rendering
        self.frames = []
    
    def set_difficulty(self, difficulty):
        """Adjust game parameters based on difficulty level"""
        if difficulty == 'easy':
            self.ball_speed_multiplier = 0.8
            self.paddle_speed_multiplier = 1.2
            self.noise_factor = 0.05
        elif difficulty == 'medium':
            self.ball_speed_multiplier = 1.0
            self.paddle_speed_multiplier = 1.0
            self.noise_factor = 0.1
        elif difficulty == 'hard':
            self.ball_speed_multiplier = 1.3
            self.paddle_speed_multiplier = 0.8
            self.noise_factor = 0.2
        elif difficulty == 'expert':
            self.ball_speed_multiplier = 1.5
            self.paddle_speed_multiplier = 0.7
            self.noise_factor = 0.3
        
    def reset_game(self):
        # Ball position at center
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        
        # Random initial ball direction with difficulty-based speed
        self.ball_speed_x = self.initial_ball_speed_x * self.ball_speed_multiplier * np.random.choice([-1, 1])
        self.ball_speed_y = self.initial_ball_speed_y * self.ball_speed_multiplier * np.random.choice([-1, 1])
        
        # Paddles position, centered vertically
        self.paddle_left_y = (self.height - self.paddle_height) // 2
        self.paddle_right_y = (self.height - self.paddle_height) // 2
        
        # Scoring
        self.score_left = 0
        self.score_right = 0
        
        # Game history for LSTM and CNN
        self.state_history = []
        
        # Performance metrics
        self.left_paddle_hits = 0
        self.right_paddle_hits = 0
        self.left_paddle_misses = 0
        self.right_paddle_misses = 0
        
    def update(self, left_paddle_ai=None, right_paddle_ai=None):
        # Store current state for history
        current_state = self.get_state()
        self.state_history.append(current_state)
        if len(self.state_history) > 10:  # Keep only last 10 frames
            self.state_history.pop(0)
        
        # Update ball position with added noise for realism
        noise_x = np.random.normal(0, self.noise_factor)
        noise_y = np.random.normal(0, self.noise_factor)
        self.ball_x += self.ball_speed_x + noise_x
        self.ball_y += self.ball_speed_y + noise_y
        
        # Move left paddle with AI if provided
        if left_paddle_ai is not None:
            target_y = left_paddle_ai(self)
            self.move_paddle_left(target_y)
        
        # Move right paddle with AI if provided
        if right_paddle_ai is not None:
            target_y = right_paddle_ai(self)
            self.move_paddle_right(target_y)
        
        # Ball collision with top and bottom walls
        if self.ball_y <= self.ball_radius or self.ball_y >= self.height - self.ball_radius:
            self.ball_speed_y *= -1
        
        # Ball collision with paddles
        # Left paddle
        if (self.ball_x - self.ball_radius <= self.paddle_width and 
            self.paddle_left_y <= self.ball_y <= self.paddle_left_y + self.paddle_height):
            self.ball_speed_x *= -1
            self.left_paddle_hits += 1
            # Add a bit of randomness to the bounce
            self.ball_speed_y += np.random.uniform(-1, 1)
        
        # Right paddle
        if (self.ball_x + self.ball_radius >= self.width - self.paddle_width and 
            self.paddle_right_y <= self.ball_y <= self.paddle_right_y + self.paddle_height):
            self.ball_speed_x *= -1
            self.right_paddle_hits += 1
            # Add a bit of randomness to the bounce
            self.ball_speed_y += np.random.uniform(-1, 1)
        
        # Scoring - ball goes past paddles
        if self.ball_x < 0:
            self.score_right += 1
            self.left_paddle_misses += 1
            self.reset_ball()
        
        if self.ball_x > self.width:
            self.score_left += 1
            self.right_paddle_misses += 1
            self.reset_ball()
    
    def reset_ball(self):
        """Reset ball to center with random direction"""
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_speed_x = self.initial_ball_speed_x * self.ball_speed_multiplier * np.random.choice([-1, 1])
        self.ball_speed_y = self.initial_ball_speed_y * self.ball_speed_multiplier * np.random.choice([-1, 1])
    
    def move_paddle_left(self, target_y):
        """Move left paddle toward target y position with difficulty-based speed"""
        paddle_speed = self.paddle_speed * self.paddle_speed_multiplier
        if target_y < self.paddle_left_y:
            self.paddle_left_y = max(0, self.paddle_left_y - paddle_speed)
        elif target_y > self.paddle_left_y:
            self.paddle_left_y = min(self.height - self.paddle_height, self.paddle_left_y + paddle_speed)
    
    def move_paddle_right(self, target_y):
        """Move right paddle toward target y position with difficulty-based speed"""
        paddle_speed = self.paddle_speed * self.paddle_speed_multiplier
        if target_y < self.paddle_right_y:
            self.paddle_right_y = max(0, self.paddle_right_y - paddle_speed)
        elif target_y > self.paddle_right_y:
            self.paddle_right_y = min(self.height - self.paddle_height, self.paddle_right_y + paddle_speed)
    
    def get_state(self):
        """Return the current game state as a dictionary"""
        return {
            'ball_x': self.ball_x,
            'ball_y': self.ball_y,
            'ball_speed_x': self.ball_speed_x,
            'ball_speed_y': self.ball_speed_y,
            'paddle_left_y': self.paddle_left_y,
            'paddle_right_y': self.paddle_right_y
        }
        
    def get_state_history(self):
        """Return state history for temporal models"""
        return self.state_history
        
    def get_image_state(self, size=(84, 84)):
        """Generate a simple image representation of the game state"""
        # Create a blank image (black background)
        image = np.zeros((*size, 1), dtype=np.float32)
        
        # Scale coordinates to image size
        scale_x = size[1] / self.width
        scale_y = size[0] / self.height
        
        # Draw ball (white dot)
        ball_x = int(self.ball_x * scale_x)
        ball_y = int(self.ball_y * scale_y)
        ball_radius = max(1, int(self.ball_radius * scale_y))
        
        # Simple circle drawing
        for i in range(-ball_radius, ball_radius + 1):
            for j in range(-ball_radius, ball_radius + 1):
                if i*i + j*j <= ball_radius*ball_radius:
                    y, x = ball_y + i, ball_x + j
                    if 0 <= y < size[0] and 0 <= x < size[1]:
                        image[y, x] = 1.0
        
        # Draw paddles (white rectangles)
        paddle_width = max(1, int(self.paddle_width * scale_x))
        paddle_height = int(self.paddle_height * scale_y)
        
        # Left paddle
        left_y = int(self.paddle_left_y * scale_y)
        for y in range(left_y, min(left_y + paddle_height, size[0])):
            for x in range(paddle_width):
                image[y, x] = 1.0
        
        # Right paddle
        right_y = int(self.paddle_right_y * scale_y)
        for y in range(right_y, min(right_y + paddle_height, size[0])):
            for x in range(size[1] - paddle_width, size[1]):
                image[y, x] = 1.0
                
        return image
    
    def get_hit_rates(self):
        """Calculate hit rates for both paddles"""
        left_attempts = self.left_paddle_hits + self.left_paddle_misses
        right_attempts = self.right_paddle_hits + self.right_paddle_misses
        
        left_hit_rate = self.left_paddle_hits / left_attempts if left_attempts > 0 else 0
        right_hit_rate = self.right_paddle_hits / right_attempts if right_attempts > 0 else 0
        
        return {
            'left_hit_rate': left_hit_rate,
            'right_hit_rate': right_hit_rate,
            'left_hits': self.left_paddle_hits,
            'left_misses': self.left_paddle_misses,
            'right_hits': self.right_paddle_hits,
            'right_misses': self.right_paddle_misses
        }

    def get_optimal_paddle_position(self, is_left_paddle=True):
        """Calculate optimal paddle position based on ball trajectory"""
        if (is_left_paddle and self.ball_speed_x < 0) or (not is_left_paddle and self.ball_speed_x > 0):
            # Ball is moving toward this paddle
            # Calculate where ball will intersect paddle plane
            if is_left_paddle:
                intersection_x = self.paddle_width
                time_to_intersect = (intersection_x - self.ball_x) / self.ball_speed_x
            else:
                intersection_x = self.width - self.paddle_width
                time_to_intersect = (intersection_x - self.ball_x) / self.ball_speed_x
            
            # Predict y position at intersection
            if time_to_intersect > 0:
                intersection_y = self.ball_y + self.ball_speed_y * time_to_intersect
                
                # Handle bounces off top/bottom walls
                while intersection_y < 0 or intersection_y > self.height:
                    if intersection_y < 0:
                        intersection_y = -intersection_y  # Bounce off top
                    elif intersection_y > self.height:
                        intersection_y = 2 * self.height - intersection_y  # Bounce off bottom
                
                # Return optimal position to center paddle on ball
                return intersection_y - self.paddle_height / 2
            
        # Default: center the paddle on the ball's current y-position
        return self.ball_y - self.paddle_height / 2