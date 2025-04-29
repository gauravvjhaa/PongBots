import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from tqdm import tqdm
import itertools
import warnings
from pong_simulator import PongSimulator

# Suppress StandardScaler warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

# Create necessary directories
os.makedirs('competition/results', exist_ok=True)
os.makedirs('competition/visualizations', exist_ok=True)

# Current date and user info
CURRENT_DATE = "2025-04-29 12:37:03"
CURRENT_USER = "gauravvjhaastuck"

def prepare_tabular_features(state):
    """Prepare tabular features from a game state"""
    is_ball_moving_right = 1 if state['ball_speed_x'] > 0 else 0
    is_ball_moving_down = 1 if state['ball_speed_y'] > 0 else 0
    distance_to_left_paddle = state['ball_x']
    distance_to_right_paddle = 800 - state['ball_x']  # 800 is the screen width
    ball_speed = np.sqrt(state['ball_speed_x']**2 + state['ball_speed_y']**2)
    
    return [
        state['ball_x'],
        state['ball_y'],
        state['ball_speed_x'],
        state['ball_speed_y'],
        is_ball_moving_right,
        is_ball_moving_down,
        distance_to_left_paddle,
        distance_to_right_paddle,
        ball_speed
    ]

def prepare_sequence_data(history):
    """Prepare sequence data for LSTM model"""
    sequence = []
    for past_state in history:
        features = [
            past_state['ball_x'], past_state['ball_y'],
            past_state['ball_speed_x'], past_state['ball_speed_y'],
            past_state['paddle_left_y'], past_state['paddle_right_y']
        ]
        sequence.append(features)
    return np.array([sequence])  # Add batch dimension

def get_model_prediction(model_info, simulator):
    """Get paddle position prediction from a model"""
    model_type = model_info['type']
    model = model_info['model']
    scaler = model_info.get('scaler')
    
    state = simulator.get_state()
    
    if model_type == 'SimpleFF' or model_type == 'DeepFF':
        # Prepare tabular features
        input_data = prepare_tabular_features(state)
        if scaler:
            # Convert to DataFrame with proper column names to avoid warning
            column_names = [
                'ball_x', 'ball_y', 'ball_speed_x', 'ball_speed_y',
                'is_ball_moving_right', 'is_ball_moving_down',
                'distance_to_left_paddle', 'distance_to_right_paddle',
                'ball_speed'
            ]
            input_df = pd.DataFrame([input_data], columns=column_names)
            input_scaled = scaler.transform(input_df)
            return model.predict(input_scaled, verbose=0)[0][0]
        else:
            return model.predict(np.array([input_data]), verbose=0)[0][0]
            
    elif model_type == 'LSTM':
        # Get sequence history
        history = simulator.get_state_history()
        if len(history) < 10:  # Not enough history yet
            return simulator.height / 2  # Default to middle
            
        # Prepare sequence data
        sequence = prepare_sequence_data(history[-10:])
        return model.predict(sequence, verbose=0)[0][0]
    
    # Fallback
    return simulator.height / 2

def run_match(model1_info, model2_info, n_points=3, max_steps=1000):
    """Run a match between two models until one scores n_points"""
    simulator = PongSimulator(difficulty='hard')  # Increased difficulty for faster scoring
    simulator.reset_game()
    
    step_count = 0
    
    # Run until one model scores n points or max steps reached
    while (simulator.score_left < n_points and simulator.score_right < n_points) and step_count < max_steps:
        # Get predictions from both models
        left_position = get_model_prediction(model1_info, simulator)
        right_position = get_model_prediction(model2_info, simulator)
        
        # Define paddle AIs for simulation
        def left_paddle_ai(sim):
            return left_position
        
        def right_paddle_ai(sim):
            return right_position
        
        # Update game state
        simulator.update(left_paddle_ai=left_paddle_ai, right_paddle_ai=right_paddle_ai)
        step_count += 1
    
    # Record final hit rates
    hit_rates = simulator.get_hit_rates()
    
    # Determine winner
    if simulator.score_left > simulator.score_right:
        winner = model1_info['type']
    elif simulator.score_right > simulator.score_left:
        winner = model2_info['type']
    else:
        winner = 'Draw'
    
    # Compile match results
    match_result = {
        'model1': model1_info['type'],
        'model2': model2_info['type'],
        'score_model1': simulator.score_left,
        'score_model2': simulator.score_right,
        'hit_rate_model1': hit_rates['left_hit_rate'],
        'hit_rate_model2': hit_rates['right_hit_rate'],
        'hits_model1': hit_rates['left_hits'],
        'hits_model2': hit_rates['right_hits'],
        'misses_model1': hit_rates['left_misses'],
        'misses_model2': hit_rates['right_misses'],
        'steps': step_count,
        'winner': winner
    }
    
    return match_result

def run_competitions(n_matches=20):
    """Run competitions between all model pairs"""
    print("Loading models...")
    
    # Load models
    try:
        # Try to load scaler for tabular data
        with open('data/processed/tabular_metadata.pkl', 'rb') as f:
            scaler, _ = pickle.load(f)
    except:
        print("Scaler not found. Will use unscaled inputs for tabular models.")
        scaler = None
    
    # Model paths - using the paths you specified
    model_paths = {
        'SimpleFF': '/kaggle/working/models/simple_ff_best.keras',
        'DeepFF': '/kaggle/working/models/deep_ff_best.keras',
        'LSTM': '/kaggle/working/models/lstm_best.keras',
    }
    
    # Load models
    models = {}
    for model_type, path in model_paths.items():
        try:
            models[model_type] = {
                'type': model_type,
                'model': load_model(path),
                'scaler': scaler if model_type in ['SimpleFF', 'DeepFF'] else None
            }
            print(f"Successfully loaded {model_type} model from {path}")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
    
    # Define model pairs for 1v1 competitions
    model_pairs = list(itertools.combinations(models.keys(), 2))
    
    # Track all competition results
    all_results = []
    summary_results = []
    
    # Run competitions for each pair
    for model1_type, model2_type in model_pairs:
        print(f"\nRunning competition: {model1_type} vs {model2_type} ({n_matches} matches)")
        model1_info = models[model1_type]
        model2_info = models[model2_type]
        
        # Track results for this matchup
        model1_wins = 0
        model2_wins = 0
        draws = 0
        model1_total_hit_rate = 0
        model2_total_hit_rate = 0
        model1_points = 0
        model2_points = 0
        total_steps = 0
        
        # Run all matches
        for match_num in tqdm(range(n_matches)):
            # Run match
            match_result = run_match(model1_info, model2_info)
            
            # Track statistics
            if match_result['winner'] == model1_type:
                model1_wins += 1
            elif match_result['winner'] == model2_type:
                model2_wins += 1
            else:
                draws += 1
                
            model1_total_hit_rate += match_result['hit_rate_model1']
            model2_total_hit_rate += match_result['hit_rate_model2']
            model1_points += match_result['score_model1']
            model2_points += match_result['score_model2']
            total_steps += match_result['steps']
            
            # Add detailed result
            all_results.append({
                'match_id': f"{model1_type}_vs_{model2_type}_{match_num}",
                'model1': model1_type,
                'model2': model2_type, 
                'score': f"{match_result['score_model1']}-{match_result['score_model2']}",
                'winner': match_result['winner'],
                'hit_rate_model1': match_result['hit_rate_model1'],
                'hit_rate_model2': match_result['hit_rate_model2'],
                'steps': match_result['steps']
            })
        
        # Calculate averages
        model1_avg_hit_rate = model1_total_hit_rate / n_matches
        model2_avg_hit_rate = model2_total_hit_rate / n_matches
        avg_steps_per_match = total_steps / n_matches
        
        # Add summary result
        summary_results.append({
            'matchup': f"{model1_type} vs {model2_type}",
            'model1': model1_type,
            'model2': model2_type,
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'draws': draws,
            'model1_win_rate': model1_wins / n_matches,
            'model2_win_rate': model2_wins / n_matches,
            'model1_avg_hit_rate': model1_avg_hit_rate,
            'model2_avg_hit_rate': model2_avg_hit_rate,
            'model1_points': model1_points,
            'model2_points': model2_points,
            'avg_steps_per_match': avg_steps_per_match
        })
        
        print(f"Results: {model1_type} ({model1_wins} wins) - {model2_type} ({model2_wins} wins) - {draws} draws")
        print(f"Average hit rates: {model1_type}: {model1_avg_hit_rate:.4f}, {model2_type}: {model2_avg_hit_rate:.4f}")
    
    # Save detailed results
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv('competition/results/detailed_results.csv', index=False)
    
    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('competition/results/summary_results.csv', index=False)
    
    print("\nCompetition complete!")
    print(f"Detailed results saved to competition/results/detailed_results.csv")
    print(f"Summary results saved to competition/results/summary_results.csv")
    
    return all_results, summary_results

def calculate_rankings(summary_results):
    """Calculate overall rankings based on competition results"""
    # Identify all models
    all_models = set()
    for result in summary_results:
        all_models.add(result['model1'])
        all_models.add(result['model2'])
    
    # Initialize ranking data
    rankings = {model: {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'points_scored': 0,
        'points_conceded': 0,
        'total_hit_rate': 0.0,
        'matches': 0
    } for model in all_models}
    
    # Calculate stats for each model
    for result in summary_results:
        model1, model2 = result['model1'], result['model2']
        
        # Wins/losses/draws
        rankings[model1]['wins'] += result['model1_wins']
        rankings[model2]['wins'] += result['model2_wins']
        rankings[model1]['losses'] += result['model2_wins']
        rankings[model2]['losses'] += result['model1_wins']
        rankings[model1]['draws'] += result['draws']
        rankings[model2]['draws'] += result['draws']
        
        # Points
        rankings[model1]['points_scored'] += result['model1_points']
        rankings[model2]['points_scored'] += result['model2_points']
        rankings[model1]['points_conceded'] += result['model2_points']
        rankings[model2]['points_conceded'] += result['model1_points']
        
        # Hit rates
        matches = result['model1_wins'] + result['model2_wins'] + result['draws']
        rankings[model1]['total_hit_rate'] += result['model1_avg_hit_rate'] * matches
        rankings[model2]['total_hit_rate'] += result['model2_avg_hit_rate'] * matches
        rankings[model1]['matches'] += matches
        rankings[model2]['matches'] += matches
    
    # Calculate final metrics
    ranking_data = []
    for model, stats in rankings.items():
        matches_played = stats['wins'] + stats['losses'] + stats['draws']
        win_rate = stats['wins'] / matches_played if matches_played > 0 else 0
        avg_hit_rate = stats['total_hit_rate'] / stats['matches'] if stats['matches'] > 0 else 0
        
        ranking_data.append({
            'model': model,
            'wins': stats['wins'],
            'losses': stats['losses'],
            'draws': stats['draws'],
            'win_rate': win_rate,
            'avg_hit_rate': avg_hit_rate,
            'points_scored': stats['points_scored'],
            'points_conceded': stats['points_conceded'],
            'point_difference': stats['points_scored'] - stats['points_conceded'],
            'score': (win_rate * 5) + (avg_hit_rate * 2) + (stats['points_scored'] / 500)
        })
    
    # Sort by composite score
    ranking_data.sort(key=lambda x: x['score'], reverse=True)
    
    # Add rank
    for i, data in enumerate(ranking_data):
        data['rank'] = i + 1
    
    return ranking_data

def visualize_results(all_results, summary_results, rankings):
    """Create visualizations from competition results"""
    # 1. Win rates bar chart
    plt.figure(figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    matchups = [r['matchup'] for r in summary_results]
    model1_win_rates = [r['model1_win_rate'] for r in summary_results]
    model2_win_rates = [r['model2_win_rate'] for r in summary_results]
    draw_rates = [r['draws']/len(all_results)*3 for r in summary_results]
    
    bar_width = 0.25
    x = np.arange(len(matchups))
    
    plt.bar(x - bar_width, model1_win_rates, bar_width, label='Model 1 Wins', color='blue')
    plt.bar(x, draw_rates, bar_width, label='Draws', color='gray')
    plt.bar(x + bar_width, model2_win_rates, bar_width, label='Model 2 Wins', color='red')
    
    plt.xlabel('Matchup')
    plt.ylabel('Win Rate')
    plt.title('Win Rates in Model vs Model Competitions')
    plt.xticks(x, matchups)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)
    
    # Add percentages on top of bars
    for i, v in enumerate(model1_win_rates):
        plt.text(i - bar_width, v + 0.02, f"{v:.0%}", ha='center')
    
    for i, v in enumerate(draw_rates):
        if v > 0.05:  # Only add text if there's enough space
            plt.text(i, v + 0.02, f"{v:.0%}", ha='center')
    
    for i, v in enumerate(model2_win_rates):
        plt.text(i + bar_width, v + 0.02, f"{v:.0%}", ha='center')
    
    plt.tight_layout()
    plt.savefig('competition/visualizations/win_rates.png')
    
    # 2. Hit rate comparison
    plt.figure(figsize=(12, 8))
    
    for i, result in enumerate(summary_results):
        plt.scatter(i-0.1, result['model1_avg_hit_rate'], color='blue', s=100, label=result['model1'] if i==0 else "")
        plt.scatter(i+0.1, result['model2_avg_hit_rate'], color='red', s=100, label=result['model2'] if i==0 else "")
        
        # Connect the points
        plt.plot([i-0.1, i+0.1], [result['model1_avg_hit_rate'], result['model2_avg_hit_rate']], 'k--', alpha=0.5)
        
        # Add labels
        plt.text(i-0.1, result['model1_avg_hit_rate']-0.03, f"{result['model1_avg_hit_rate']:.4f}", ha='center')
        plt.text(i+0.1, result['model2_avg_hit_rate']-0.03, f"{result['model2_avg_hit_rate']:.4f}", ha='center')
    
    plt.xticks(range(len(summary_results)), [r['matchup'] for r in summary_results])
    plt.ylabel('Average Hit Rate')
    plt.title('Hit Rate Comparison by Matchup')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('competition/visualizations/hit_rates.png')
    
    # 3. Overall model rankings
    plt.figure(figsize=(14, 10))
    
    # Sort models by rank
    ranking_df = pd.DataFrame(rankings).sort_values('rank')
    
    # Create a horizontal bar chart
    bars = plt.barh(ranking_df['model'], ranking_df['score'], color=sns.color_palette('viridis', len(rankings)))
    
    plt.xlabel('Composite Score')
    plt.title('Overall Model Rankings')
    plt.gca().invert_yaxis()  # Highest rank at the top
    
    # Add score labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f"{width:.2f}", ha='left', va='center')
    
    # Add more information in a table
    plt.table(cellText=ranking_df[['rank', 'wins', 'losses', 'draws', 'win_rate', 'avg_hit_rate']].values,
              colLabels=['Rank', 'Wins', 'Losses', 'Draws', 'Win Rate', 'Hit Rate'],
              rowLabels=ranking_df['model'].values,
              cellLoc='center',
              loc='bottom', bbox=[0, -0.6, 1, 0.4])
    
    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.savefig('competition/visualizations/rankings.png')
    
    # 4. Average steps per match
    plt.figure(figsize=(10, 6))
    avg_steps = [r['avg_steps_per_match'] for r in summary_results]
    plt.bar(matchups, avg_steps, color='purple')
    
    plt.xlabel('Matchup')
    plt.ylabel('Average Steps')
    plt.title('Average Match Length by Matchup')
    plt.xticks(rotation=45)
    
    # Add values on top of bars
    for i, v in enumerate(avg_steps):
        plt.text(i, v + 10, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('competition/visualizations/avg_steps.png')
    
    print("Visualizations saved to competition/visualizations/")

def generate_report(summary_results, rankings):
    """Generate a markdown report of competition results"""
    report = "# Neural Network Architecture Competition Report\n\n"
    report += f"Generated on: {CURRENT_DATE} UTC\n"
    report += f"Generated by: {CURRENT_USER}\n\n"
    report += "## Competition Overview\n\n"
    report += "This report presents the results of head-to-head competitions between three neural network architectures "
    report += "for Pong paddle control. Each model played 20 matches against each other model.\n\n"
    
    # Overall rankings
    report += "## Overall Rankings\n\n"
    report += "| Rank | Model | Wins | Losses | Win Rate | Hit Rate | Points Scored | Points Conceded | Score |\n"
    report += "|------|-------|------|--------|----------|----------|---------------|----------------|-------|\n"
    
    for rank in rankings:
        report += f"| {rank['rank']} | {rank['model']} | {rank['wins']} | {rank['losses']} | "
        report += f"{rank['win_rate']:.2%} | {rank['avg_hit_rate']:.4f} | {rank['points_scored']} | "
        report += f"{rank['points_conceded']} | {rank['score']:.2f} |\n"
    
    # Head-to-head results
    report += "\n## Head-to-Head Results\n\n"
    report += "| Matchup | Model 1 Wins | Model 2 Wins | Draws | Model 1 Hit Rate | Model 2 Hit Rate | Average Match Length |\n"
    report += "|---------|--------------|--------------|-------|------------------|------------------|----------------------|\n"
    
    for result in summary_results:
        report += f"| {result['matchup']} | {result['model1_wins']} | {result['model2_wins']} | {result['draws']} | "
        report += f"{result['model1_avg_hit_rate']:.4f} | {result['model2_avg_hit_rate']:.4f} | "
        report += f"{result['avg_steps_per_match']:.1f} steps |\n"
    
    # Key findings
    report += "\n## Key Findings\n\n"
    
    # Best model
    best_model = rankings[0]['model']
    best_win_rate = rankings[0]['win_rate']
    best_hit_rate = rankings[0]['avg_hit_rate']
    
    report += f"### Best Overall Model: {best_model}\n\n"
    report += f"The {best_model} model demonstrated superior performance with a win rate of {best_win_rate:.2%} "
    report += f"and an average hit rate of {best_hit_rate:.4f}. "
    
    # Compare hit rates
    report += "\n### Hit Rate Comparison\n\n"
    hit_rates = {r['model']: r['avg_hit_rate'] for r in rankings}
    sorted_hit_rates = sorted(hit_rates.items(), key=lambda x: x[1], reverse=True)
    
    report += "Models ranked by hit rate effectiveness:\n\n"
    for i, (model, rate) in enumerate(sorted_hit_rates):
        report += f"{i+1}. **{model}**: {rate:.4f}\n"
    
    # Model vs model insights
    report += "\n### Model Matchup Insights\n\n"
    
    for result in summary_results:
        model1 = result['model1']
        model2 = result['model2']
        
        # Determine if there's a clear winner
        if result['model1_wins'] > 1.5 * result['model2_wins']:
            report += f"- **{model1}** strongly outperformed **{model2}** with {result['model1_wins']} wins vs {result['model2_wins']} losses.\n"
        elif result['model2_wins'] > 1.5 * result['model1_wins']:
            report += f"- **{model2}** strongly outperformed **{model1}** with {result['model2_wins']} wins vs {result['model1_wins']} losses.\n"
        else:
            report += f"- The matchup between **{model1}** and **{model2}** was closely contested with results of {result['model1_wins']}-{result['model2_wins']}-{result['draws']}.\n"
    
    # Conclusion
    report += "\n## Conclusion\n\n"
    report += f"Based on the comprehensive evaluation, the **{best_model}** architecture demonstrates the best "
    report += "overall performance for Pong paddle control. This suggests that this neural network architecture "
    report += "provides the optimal balance of paddle positioning accuracy and game-winning capability.\n\n"
    
    # Write report to file
    with open('competition/competition_report.md', 'w') as f:
        f.write(report)
    
    print(f"Competition report generated: competition/competition_report.md")
    
    return report

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Starting model competitions...")
    print(f"Current date/time: {CURRENT_DATE}")
    print(f"User: {CURRENT_USER}")
    
    # Run all competitions with fewer matches
    all_results, summary_results = run_competitions(n_matches=20)
    
    # Calculate rankings
    rankings = calculate_rankings(summary_results)
    
    # Create visualizations
    visualize_results(all_results, summary_results, rankings)
    
    # Generate report
    generate_report(summary_results, rankings)
    
    print("Competition analysis complete!")
