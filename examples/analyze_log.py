import argparse
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def parse_blocks(log_lines):
    """
    Parses log file lines to extract collision information by iteration.
    Returns a list of (iteration, block_lines) tuples.
    """
    blocks = []
    current_block = []
    current_iter = None
    in_result_block = False
    
    for line in log_lines:
        # Check for iteration information in status lines
        if "|" in line and "iter" in line and not line.startswith("Trial name"):
            try:
                cols = line.split("|")
                if len(cols) >= 4:
                    iter_num = int(cols[4].strip())
                    if current_iter is not None and current_block:
                        blocks.append((current_iter, current_block))
                    current_iter = iter_num
                    current_block = []
                    in_result_block = True
            except (ValueError, IndexError):
                continue
        
        # Check for collision information
        if "Collision detected" in line and in_result_block:
            current_block.append(line)
            
        # Check for end of result block
        if "== Status ==" in line:
            if current_iter is not None and current_block:
                blocks.append((current_iter, current_block))
            in_result_block = False
    
    # Add the last block
    if current_iter is not None and current_block:
        blocks.append((current_iter, current_block))
    
    return blocks

def count_collisions_in_block(block_lines):
    """
    Counts collision occurrences in a block.
    """
    return len([line for line in block_lines if "Collision detected" in line])

def plot_collision_history(iterations, collision_counts, output_file="collision_history.png"):
    """
    Creates and saves a plot of collision history.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, collision_counts, marker='o', linestyle='-', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Collisions')
    plt.title('Collision History by Iteration')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Analyzes collision occurrences in training log file."
    )
    parser.add_argument("logfile", help="Path to the log file")
    parser.add_argument("--plot", action="store_true", help="Generate collision history plot")
    args = parser.parse_args()

    try:
        with open(args.logfile, "r", encoding='utf-8', errors='ignore') as f:
            log_lines = f.readlines()
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    blocks = parse_blocks(log_lines)
    
    print("\n=== Collision Analysis Summary ===")
    iterations = []
    collision_counts = []
    
    for iteration, block_lines in sorted(blocks):
        collision_count = count_collisions_in_block(block_lines)
        if collision_count > 0:
            print(f"Iteration {iteration}: {collision_count} collisions")
        iterations.append(iteration)
        collision_counts.append(collision_count)

    if args.plot and iterations:
        try:
            plot_collision_history(iterations, collision_counts)
            print(f"\nCollision history plot saved as 'collision_history.png'")
        except Exception as e:
            print(f"\nError generating plot: {e}")

    # Print summary statistics
    if collision_counts:
        print(f"\nSummary Statistics:")
        print(f"Total iterations analyzed: {len(iterations)}")
        print(f"Total collisions: {sum(collision_counts)}")
        print(f"Average collisions per iteration: {sum(collision_counts)/len(collision_counts):.2f}")
        print(f"Maximum collisions in one iteration: {max(collision_counts)}")

if __name__ == "__main__":
    main()