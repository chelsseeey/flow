import argparse
import re
import matplotlib.pyplot as plt

def parse_blocks(log_lines):
    """
    Parses log file lines, extracting each iteration block.
    Each block starts with "Result for", and "training_iteration:" is searched within the block
    to record the iteration number.
    Returns a list of (iteration, block_lines) tuples for each block.
    If the iteration number is not found, it is marked as None.
    """
    blocks = []
    current_block = []
    current_iter = 1  # Start with iteration 1
    first_table_found = False
    
    for i, line in enumerate(log_lines):
        # Check for first status table
        if not first_table_found and "+-----------------------------------------+" in line:
            first_table_found = True
            blocks.append((current_iter, current_block))
            current_block = []
            
            # Get iteration number from the table
            try:
                table_lines = log_lines[i:i+4]
                for table_line in table_lines:
                    if "|" in table_line and "iter" in table_line:
                        current_iter = int(table_line.split("|")[4].strip())
                        break
            except (ValueError, IndexError):
                continue
                
        # For subsequent iterations
        elif first_table_found and "+-----------------------------------------+" in line:
            try:
                table_lines = log_lines[i:i+4]
                for table_line in table_lines:
                    if "|" in table_line and "iter" in table_line:
                        iter_num = int(table_line.split("|")[4].strip())
                        if current_block:
                            blocks.append((current_iter, current_block))
                        current_iter = iter_num
                        current_block = []
                        break
            except (ValueError, IndexError):
                continue
        
        # Collect collision information
        elif "Collision detected" in line:
            current_block.append(line)
    
    # Add the last block
    if current_block:
        blocks.append((current_iter, current_block))
    
    return blocks

def count_collisions_in_block(block_lines):
    """
    Counts the number of "Collision detected at time step" occurrences within the given block.
    """
    return sum(1 for line in block_lines if "Collision detected at time step" in line)

def main():
    parser = argparse.ArgumentParser(
        description="Calculates the number of 'Collision detected at time step' occurrences per iteration in a log file."
    )
    parser.add_argument("logfile", help="Path to the log file")
    args = parser.parse_args()

    try:
        # 먼저 'utf-8'로 시도
        try:
            with open(args.logfile, "r", encoding='utf-8', errors='ignore') as f:
                log_lines = f.readlines()
        except UnicodeDecodeError:
            # 'utf-8' 실패시 'latin-1'로 시도
            with open(args.logfile, "r", encoding='latin-1') as f:
                log_lines = f.readlines()
    except Exception as e:
        print(f"Could not open the log file: {e}")
        return

    blocks = parse_blocks(log_lines)

    print("=== Collision Count Summary ===")
    iterations = []
    collision_counts = []
    for iteration, block_lines in blocks:
        # Output 'Unknown' if there is no iteration number
        iteration_label = str(iteration) if iteration is not None else "Unknown"
        collision_count = count_collisions_in_block(block_lines)
        print(f"Iteration {iteration_label}: {collision_count} collisions")
        iterations.append(iteration_label)
        collision_counts.append(collision_count)

    # 그래프 그리기: x축은 iteration, y축은 collision 횟수
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, collision_counts, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Collision Count')
    plt.title('Collisions per Iteration')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()