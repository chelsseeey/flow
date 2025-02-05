import argparse
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Flow training results"
    )
    parser.add_argument(
        '--exp_dir',        # 실험 디렉토리 경로
        type=str,
        default="/root/ray_results/figure8_with_lights",
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--exp_id',     # 실험 ID
        type=str,
        default="PPO_MultiAgentAccelPOEnv",
        help='Experiment name prefix'
    )
    return parser.parse_args()

def analyze_training_results(exp_dir, exp_id):      # 결과 분석 함수

    # Add pandas display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', int(10000))
    results = []
    
    # Find all progress files
    for trial_dir in glob.glob(f"{exp_dir}/{exp_id}*"):
        progress_file = os.path.join(trial_dir, "progress.csv")
        
        if os.path.exists(progress_file):
            df = pd.read_csv(progress_file)
            trial_name = os.path.basename(trial_dir)
            
            # Extract metrics per iteration
            for _, row in df.iterrows():
                results.append({
                    'Trial Name': trial_name,
                    'Iteration': row['training_iteration'],
                    'Timesteps': row['timesteps_total'],
                    'Reward': row['episode_reward_mean'],
                    'Time(s)': row['time_total_s']
                })
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Print results
        print(results_df.to_string(index=False))
        
        # Create reward plot
        plt.figure(figsize=(10,6))
        plt.plot(results_df['Iteration'], results_df['Reward'], 'b-', marker='o')
        plt.title('Training Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.show()  # Display plot window
    else:
        print(f"No results found in {exp_dir}")

if __name__ == "__main__":
    args = parse_args()
    analyze_training_results(args.exp_dir, args.exp_id)