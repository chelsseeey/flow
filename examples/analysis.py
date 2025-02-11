import argparse
import pandas as pd
import glob
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from warning_logger import collision_logger

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
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training metrics
    for trial_name, group in combined_df.groupby('trial_name'):
        ax1.plot(group['training_iteration'], 
                group['episode_reward_mean'], 
                label=trial_name)
    
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Plot collision data
    try:
        collision_data = collision_logger.load_collision_data(exp_id)
        if collision_data:
            ax2.plot(range(1, len(collision_logger.iteration_collisions) + 1),
                    collision_logger.iteration_collisions, 'r-')
            ax2.set_xlabel('Iteration Number')
            ax2.set_ylabel('Number of Collisions')
            ax2.set_title(f'Collisions - {exp_id}')
            ax2.grid(True)
        else:
            print(f"No collision data found for {exp_id}")
    except Exception as e:
        print(f"Error loading collision data: {e}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    analyze_training_results(args.exp_dir, args.exp_id)