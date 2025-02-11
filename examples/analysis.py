import argparse
import pandas as pd
import glob
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from warning_logger import collision_logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Flow training results"
    )
    parser.add_argument(
        '--exp_dir',
        type=str,
        default="/root/ray_results/figure8_with_lights",
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--exp_id',
        type=str,
        default="PPO_MultiAgentAccelPOEnv",
        help='Experiment name prefix'
    )
    return parser.parse_args()

def extract_safety_metrics(row):
    events = {
        'warnings': 0,
        'collisions': 0,
        'speed_violations': 0
    }
    
    try:
        if 'sampler_perf' in row:
            sampler_perf = eval(str(row['sampler_perf']))
            for key in sampler_perf:
                if 'warning' in str(key).lower():
                    events['warnings'] += int(sampler_perf[key])
                if 'collision' in str(key).lower():
                    events['collisions'] += int(sampler_perf[key])
                if 'speed' in str(key).lower() and 'violation' in str(key).lower():
                    events['speed_violations'] += int(sampler_perf[key])
    except Exception as e:
        print(f"Error processing safety metrics: {e}")
    
    return events

def analyze_training_results(exp_dir, exp_id):
    # Find progress files
    progress_files = glob.glob(os.path.join(exp_dir, f"{exp_id}*", "progress.csv"))
    
    if not progress_files:
        print(f"No progress files found in {exp_dir} for experiment {exp_id}")
        return
    
    # Load and process data
    dfs = []
    for file in progress_files:
        trial_name = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        df['trial_name'] = trial_name
        dfs.append(df)
    
    combined_df = pd.concat(dfs)
    
    # Plot training metrics
    plt.figure(figsize=(12, 6))
    for trial_name, group in combined_df.groupby('trial_name'):
        plt.plot(group['training_iteration'], 
                group['episode_reward_mean'], 
                label=trial_name)
    
    plt.xlabel('Training Iteration')
    plt.ylabel('Mean Episode Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot collision data
    for trial_name in combined_df['trial_name'].unique():
        collision_logger.plot_collisions(trial_name)

if __name__ == "__main__":
    args = parse_args()
    analyze_training_results(args.exp_dir, args.exp_id)