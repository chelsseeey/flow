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
    current_block_lines = []
    current_iteration = None

    for line in log_lines:
        # Start of a new block: lines starting with "Result for"
        if line.startswith("Result for"):
            # Save the existing block if it's in progress
            if current_block_lines:
                blocks.append((current_iteration, current_block_lines))
            # Initialize a new block
            current_block_lines = [line]
            current_iteration = None
        else:
            # Record only if the current block is in progress
            if current_block_lines:
                current_block_lines.append(line)
                # Find training_iteration information within the block if not already found
                if current_iteration is None and "training_iteration:" in line:
                    m = re.search(r"training_iteration:\s*(\d+)", line)
                    if m:
                        current_iteration = int(m.group(1))
    # Save the last block
    if current_block_lines:
        blocks.append((current_iteration, current_block_lines))
    return blocks

def count_collisions_in_block(block_lines):
    return sum(1 for line in block_lines if "Collision detected at time step" in line)

def main():
    parser = argparse.ArgumentParser(
        description="Calculates the number of 'Collision detected at time step' occurrences per iteration in a log file."
    )
    parser.add_argument("logfile", help="Path to the log file")
    args = parser.parse_args()

    try:
        with open(args.logfile, "r") as f:
            log_lines = f.readlines()
    except Exception as e:
        print(f"Could not open the log file: {e}")
        return

    blocks = parse_blocks(log_lines)
    if not blocks:
        print("No iteration blocks found in the log file.")
        return

    print("=== 충돌 횟수 요약 ===")
    iterations = []
    collision_counts = []

    for iteration, block_lines in blocks:
        # Output 'Unknown' if there is no iteration number
        iteration_label = str(iteration) if iteration is not None else "Unknown"
        collision_count = count_collisions_in_block(block_lines)
        print(f"Iteration {iteration_label}: {collision_count} collisions")  # 기존 로그 출력 유지
        iterations.append(iteration_label)
        collision_counts.append(collision_count)

    # Generate the graph
    plt.figure(figsize=(15, 7))
    plt.bar(range(len(iterations)), collision_counts, color='skyblue')
    plt.xlabel("Iteration")
    plt.ylabel("Number of Collisions")
    plt.title("Collision Count per Iteration")
    plt.xticks(range(len(iterations)), iterations, rotation=90)  # Show all iteration labels
    plt.tight_layout()

    # Show the graph
    plt.show()