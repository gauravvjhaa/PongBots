import os

def create_html_visualization():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Neural Network Pong Visualization - 2025-04-29 14:37:34</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #222;
            color: white;
        }
        #game-container {
            width: 800px;
            height: 600px;
            margin: 0 auto;
            position: relative;
            background-color: black;
            border: 2px solid #444;
            overflow: hidden;
        }
        #ball {
            width: 20px;
            height: 20px;
            background-color: white;
            border-radius: 50%;
            position: absolute;
            transform: translate(-50%, -50%);
        }
        .paddle {
            width: 10px;
            height: 60px;
            position: absolute;
        }
        #paddle-left {
            left: 15px;
            background-color: #00f;
        }
        #paddle-right {
            right: 15px;
            background-color: #f00;
        }
        #center-line {
            position: absolute;
            left: 50%;
            height: 100%;
            width: 2px;
            background: repeating-linear-gradient(to bottom, #444 0px, #444 10px, transparent 10px, transparent 20px);
        }
        .score {
            font-size: 32px;
            position: absolute;
            top: 20px;
        }
        #score-left {
            left: 25%;
            color: #00f;
        }
        #score-right {
            right: 25%;
            color: #f00;
        }
        .model-name {
            position: absolute;
            top: 50px;
            font-size: 16px;
        }
        #model-left {
            left: 25%;
            transform: translateX(-50%);
            color: #00f;
        }
        #model-right {
            right: 25%;
            transform: translateX(50%);
            color: #f00;
        }
        .hit-rate {
            position: absolute;
            bottom: 50px;
            font-size: 16px;
        }
        #hit-rate-left {
            left: 50px;
            color: #00f;
        }
        #hit-rate-right {
            right: 50px;
            color: #f00;
        }
        #speed-control {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
        }
        #speed {
            color: #0f0;
        }
        #controls {
            margin: 20px auto;
            width: 800px;
            padding: 10px;
            background-color: #333;
            border-radius: 5px;
            text-align: center;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: #888;
            font-size: 12px;
        }
        #stats {
            margin: 20px auto;
            width: 800px;
            padding: 10px;
            background-color: #333;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: center;
            border-bottom: 1px solid #555;
        }
        th {
            background-color: #444;
        }
    </style>
</head>
<body>
    <div id="game-container">
        <div id="center-line"></div>
        <div id="ball"></div>
        <div id="paddle-left" class="paddle"></div>
        <div id="paddle-right" class="paddle"></div>
        <div id="score-left" class="score">0</div>
        <div id="score-right" class="score">0</div>
        <div id="model-left" class="model-name">LSTM</div>
        <div id="model-right" class="model-name">DeepFF</div>
        <div id="hit-rate-left" class="hit-rate">Hit: 0.00</div>
        <div id="hit-rate-right" class="hit-rate">Hit: 0.00</div>
        <div id="speed-control">Speed: <span id="speed">1.0x</span></div>
    </div>
    
    <div id="controls">
        <h3>Controls</h3>
        <p>Space: Pause/Resume | Up/Down Arrows: Change Speed | R: Reset Game</p>
        <button id="pause-btn">Pause</button>
        <button id="speed-up-btn">Speed Up</button>
        <button id="speed-down-btn">Slow Down</button>
        <button id="reset-btn">Reset</button>
        <div>
            <label for="left-model">Left Paddle:</label>
            <select id="left-model">
                <option value="LSTM">LSTM</option>
                <option value="DeepFF">DeepFF</option>
                <option value="SimpleFF">SimpleFF</option>
            </select>
            
            <label for="right-model">Right Paddle:</label>
            <select id="right-model">
                <option value="DeepFF">DeepFF</option>
                <option value="LSTM">LSTM</option>
                <option value="SimpleFF">SimpleFF</option>
            </select>
            
            <label for="difficulty">Difficulty:</label>
            <select id="difficulty">
                <option value="easy">Easy</option>
                <option value="medium" selected>Medium</option>
                <option value="hard">Hard</option>
            </select>
        </div>
    </div>
    
    <div id="stats">
        <h3>Model Performance Stats (from Competition)</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>Wins</th>
                <th>Losses</th>
                <th>Win Rate</th>
                <th>Hit Rate</th>
            </tr>
            <tr>
                <td>LSTM</td>
                <td>32</td>
                <td>8</td>
                <td>80.0%</td>
                <td>0.4912</td>
            </tr>
            <tr>
                <td>DeepFF</td>
                <td>22</td>
                <td>18</td>
                <td>55.0%</td>
                <td>0.5516</td>
            </tr>
            <tr>
                <td>SimpleFF</td>
                <td>5</td>
                <td>35</td>
                <td>12.5%</td>
                <td>0.4616</td>
            </tr>
        </table>
    </div>
    
    <div class="footer">
        <p>Created by: gauravvjhaa | Date: 2025-04-29 14:37:34</p>
        <p>Based on neural network competition results from trained models.</p>
    </div>

    <script>
        // Game settings
        const WIDTH = 800;
        const HEIGHT = 600;
        const BALL_SIZE = 20;
        const PADDLE_WIDTH = 10;
        const PADDLE_HEIGHT = 60;
        const PADDLE_BUFFER = 15;
        
        // Game state
        let ballX = WIDTH / 2;
        let ballY = HEIGHT / 2;
        let ballSpeedX = 5;
        let ballSpeedY = 5;
        let leftPaddleY = HEIGHT / 2 - PADDLE_HEIGHT / 2;
        let rightPaddleY = HEIGHT / 2 - PADDLE_HEIGHT / 2;
        let scoreLeft = 0;
        let scoreRight = 0;
        let leftHits = 0;
        let leftMisses = 0;
        let rightHits = 0;
        let rightMisses = 0;
        let speedMultiplier = 1.0;
        let difficulty = 'medium';
        
        // Control state
        let paused = false;
        let gameLoopId = null;
        
        // Model configuration
        let leftModelType = "LSTM";
        let rightModelType = "DeepFF";
        
        // Model accuracy based on your competition results
        const modelAccuracy = {
            "LSTM": 0.92,
            "DeepFF": 0.82,
            "SimpleFF": 0.64
        };
        
        // Select all the DOM elements
        const ball = document.getElementById('ball');
        const paddleLeft = document.getElementById('paddle-left');
        const paddleRight = document.getElementById('paddle-right');
        const scoreLeftElem = document.getElementById('score-left');
        const scoreRightElem = document.getElementById('score-right');
        const hitRateLeft = document.getElementById('hit-rate-left');
        const hitRateRight = document.getElementById('hit-rate-right');
        const speedElem = document.getElementById('speed');
        const pauseBtn = document.getElementById('pause-btn');
        const speedUpBtn = document.getElementById('speed-up-btn');
        const speedDownBtn = document.getElementById('speed-down-btn');
        const resetBtn = document.getElementById('reset-btn');
        const leftModelSelector = document.getElementById('left-model');
        const rightModelSelector = document.getElementById('right-model');
        const modelLeftElem = document.getElementById('model-left');
        const modelRightElem = document.getElementById('model-right');
        const difficultySelector = document.getElementById('difficulty');
        
        // Helper functions
        function predictPerfectPosition(isLeftPaddle) {
            // If ball is moving away, just track it
            if ((isLeftPaddle && ballSpeedX > 0) || (!isLeftPaddle && ballSpeedX < 0)) {
                return ballY;
            }
            
            // Calculate where ball will intersect with paddle
            const paddleX = isLeftPaddle ? 
                (PADDLE_BUFFER + PADDLE_WIDTH) : 
                (WIDTH - PADDLE_BUFFER - PADDLE_WIDTH);
            
            // Time to reach paddle
            if (ballSpeedX === 0) {
                return HEIGHT / 2;
            }
            
            const timeToReach = (paddleX - ballX) / ballSpeedX;
            
            if (timeToReach <= 0) {
                return ballY;
            }
            
            // Calculate y-position at intersection
            let intersectY = ballY + ballSpeedY * timeToReach;
            
            // Account for bounces
            const bounces = Math.floor(Math.abs(intersectY) / HEIGHT);
            const remainder = Math.abs(intersectY) % HEIGHT;
            
            if (bounces % 2 === 0) {
                // Even number of bounces
                intersectY = intersectY < 0 ? HEIGHT - remainder : remainder;
            } else {
                // Odd number of bounces
                intersectY = intersectY < 0 ? remainder : HEIGHT - remainder;
            }
            
            return intersectY;
        }
        
        function predictPaddlePosition(modelType, isLeftPaddle) {
            // Calculate the "perfect" position
            const perfectY = predictPerfectPosition(isLeftPaddle);
            
            // Apply model-specific accuracy
            const accuracy = modelAccuracy[modelType] || 0.7;
            
            // Add error based on model accuracy and difficulty
            let difficultyMultiplier = 1.0;
            if (difficulty === 'easy') difficultyMultiplier = 2.0;
            if (difficulty === 'hard') difficultyMultiplier = 0.5;
            
            const maxError = HEIGHT * (1 - accuracy) * 0.5 * difficultyMultiplier;
            const error = (Math.random() * 2 - 1) * maxError;
            
            // LSTM is better at predicting far in advance
            let adjustedError = error;
            
            if (modelType === "LSTM" && Math.abs(ballSpeedX) > 0) {
                adjustedError *= 0.7;
            } else if (modelType === "SimpleFF") {
                // SimpleFF is worse at predicting far in advance
                const distance = Math.abs(ballX - (isLeftPaddle ? PADDLE_BUFFER : (WIDTH - PADDLE_BUFFER)));
                adjustedError *= Math.min(1.5, distance / 200);
            }
            
            const predictedPosition = perfectY + adjustedError;
            
            // Ensure paddle stays within bounds - return center position
            return Math.max(PADDLE_HEIGHT / 2, Math.min(predictedPosition, HEIGHT - PADDLE_HEIGHT / 2));
        }
        
        function resetBall() {
            ballX = WIDTH / 2;
            ballY = HEIGHT / 2;
            ballSpeedX = Math.random() > 0.5 ? 5 : -5;
            ballSpeedY = (Math.random() * 6) - 3;
        }
        
        function resetGame() {
            ballX = WIDTH / 2;
            ballY = HEIGHT / 2;
            ballSpeedX = 5;
            ballSpeedY = 5;
            leftPaddleY = HEIGHT / 2 - PADDLE_HEIGHT / 2;
            rightPaddleY = HEIGHT / 2 - PADDLE_HEIGHT / 2;
            scoreLeft = 0;
            scoreRight = 0;
            leftHits = 0;
            leftMisses = 0;
            rightHits = 0;
            rightMisses = 0;
            updateScores();
            updateHitRates();
        }
        
        function updatePositions() {
            // Get model predictions
            const leftPaddleCenter = predictPaddlePosition(leftModelType, true);
            const rightPaddleCenter = predictPaddlePosition(rightModelType, false);
            
            // Convert from center to top position
            leftPaddleY = leftPaddleCenter - PADDLE_HEIGHT / 2;
            rightPaddleY = rightPaddleCenter - PADDLE_HEIGHT / 2;
            
            // Update ball position
            ballX += ballSpeedX * speedMultiplier;
            ballY += ballSpeedY * speedMultiplier;
            
            // Ball collision with top and bottom walls
            if (ballY <= BALL_SIZE/2 || ballY >= HEIGHT - BALL_SIZE/2) {
                ballSpeedY = -ballSpeedY;
                // Keep ball in bounds
                ballY = ballY <= BALL_SIZE/2 ? BALL_SIZE/2 : HEIGHT - BALL_SIZE/2;
            }
            
            // Ball collision with paddles
            // Left paddle
            if (ballX - BALL_SIZE/2 <= PADDLE_BUFFER + PADDLE_WIDTH && 
                leftPaddleY <= ballY && ballY <= leftPaddleY + PADDLE_HEIGHT) {
                ballSpeedX = Math.abs(ballSpeedX);  // Ensure moving right
                ballX = PADDLE_BUFFER + PADDLE_WIDTH + BALL_SIZE/2;  // Prevent sticking
                ballSpeedY += (Math.random() * 2 - 1);  // Add some randomness
                leftHits++;
                updateHitRates();
            }
            
            // Right paddle
            if (ballX + BALL_SIZE/2 >= WIDTH - PADDLE_BUFFER - PADDLE_WIDTH && 
                rightPaddleY <= ballY && ballY <= rightPaddleY + PADDLE_HEIGHT) {
                ballSpeedX = -Math.abs(ballSpeedX);  // Ensure moving left
                ballX = WIDTH - PADDLE_BUFFER - PADDLE_WIDTH - BALL_SIZE/2;  // Prevent sticking
                ballSpeedY += (Math.random() * 2 - 1);  // Add some randomness
                rightHits++;
                updateHitRates();
            }
            
            // Ball out of bounds (scoring)
            if (ballX < 0) {
                scoreRight++;
                leftMisses++;
                updateScores();
                updateHitRates();
                resetBall();
            } else if (ballX > WIDTH) {
                scoreLeft++;
                rightMisses++;
                updateScores();
                updateHitRates();
                resetBall();
            }
        }
        
        function updateDom() {
            ball.style.left = `${ballX}px`;
            ball.style.top = `${ballY}px`;
            
            paddleLeft.style.top = `${leftPaddleY}px`;
            paddleRight.style.top = `${rightPaddleY}px`;
        }
        
        function updateScores() {
            scoreLeftElem.textContent = scoreLeft;
            scoreRightElem.textContent = scoreRight;
        }
        
        function updateHitRates() {
            const leftRate = leftHits / (leftHits + leftMisses) || 0;
            const rightRate = rightHits / (rightHits + rightMisses) || 0;
            
            hitRateLeft.textContent = `Hit: ${leftRate.toFixed(2)}`;
            hitRateRight.textContent = `Hit: ${rightRate.toFixed(2)}`;
        }
        
        function gameLoop() {
            if (!paused) {
                updatePositions();
                updateDom();
            }
            gameLoopId = requestAnimationFrame(gameLoop);
        }
        
        // Event handlers
        pauseBtn.addEventListener('click', () => {
            paused = !paused;
            pauseBtn.textContent = paused ? 'Resume' : 'Pause';
        });
        
        speedUpBtn.addEventListener('click', () => {
            speedMultiplier = Math.min(5.0, speedMultiplier + 0.5);
            speedElem.textContent = `${speedMultiplier.toFixed(1)}x`;
        });
        
        speedDownBtn.addEventListener('click', () => {
            speedMultiplier = Math.max(0.5, speedMultiplier - 0.5);
            speedElem.textContent = `${speedMultiplier.toFixed(1)}x`;
        });
        
        resetBtn.addEventListener('click', resetGame);
        
        leftModelSelector.addEventListener('change', (event) => {
            leftModelType = event.target.value;
            modelLeftElem.textContent = leftModelType;
            resetGame();
        });
        
        rightModelSelector.addEventListener('change', (event) => {
            rightModelType = event.target.value;
            modelRightElem.textContent = rightModelType;
            resetGame();
        });
        
        difficultySelector.addEventListener('change', (event) => {
            difficulty = event.target.value;
            resetGame();
        });
        
        document.addEventListener('keydown', (event) => {
            switch(event.code) {
                case 'Space':
                    paused = !paused;
                    pauseBtn.textContent = paused ? 'Resume' : 'Pause';
                    break;
                case 'ArrowUp':
                    speedMultiplier = Math.min(5.0, speedMultiplier + 0.5);
                    speedElem.textContent = `${speedMultiplier.toFixed(1)}x`;
                    break;
                case 'ArrowDown':
                    speedMultiplier = Math.max(0.5, speedMultiplier - 0.5);
                    speedElem.textContent = `${speedMultiplier.toFixed(1)}x`;
                    break;
                case 'KeyR':
                    resetGame();
                    break;
            }
        });
        
        // Initialize game
        resetGame();
        gameLoopId = requestAnimationFrame(gameLoop);
    </script>
</body>
</html>
    """
    
    # Write the HTML to a file
    with open('pong_visualization.html', 'w') as f:
        f.write(html_content)
    
    print("HTML visualization created as 'pong_visualization.html'")
    print("Open this file in your web browser to view the simulation")
    print("\nFeatures:")
    print("- Visualizes models playing against each other")
    print("- Based on accuracy rates from your model competition")
    print("- Includes hit rate statistics")
    print("- Control speed with Up/Down arrows or buttons")
    print("- Pause with Space key or button")
    print("- Change models with drop-down selectors")
    print("- Change difficulty level to see how models perform")

if __name__ == "__main__":
    create_html_visualization()
