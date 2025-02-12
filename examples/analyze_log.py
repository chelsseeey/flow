import re
import argparse

def analyze_log(log_file):
    """Analyzes the log file and returns a dictionary of iteration-wise collision counts."""
    iteration_collisions = {}
    current_iteration = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check for iteration summary
            if "Result for PPO" in line and "training_iteration" in line:
                match = re.search(r"training_iteration: (\d+)", line)
                if match:
                    current_iteration = int(match.group(1))
                    iteration_collisions[current_iteration] = 0  # Initialize count for this iteration
            
            # Check for collision detection message
            if "Collision detected" in line and current_iteration is not None:
                iteration_collisions[current_iteration] += 1
    
    return iteration_collisions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze log file for collision counts.")
    parser.add_argument("log_file_path", type=str, help="Path to the log file")
    args = parser.parse_args()
    
    log_file_path = args.log_file_path
    collisions = analyze_log(log_file_path)
    
    print("Iteration-wise Collision Counts:")
    for iteration, count in collisions.items():
        print(f"Iteration {iteration}: {count} collisions")