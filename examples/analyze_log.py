import argparse
import re
import matplotlib.pyplot as plt

def parse_blocks(log_lines):
    """
    Parses log file lines by using lines starting with "Result for" as the end marker for each block.
    The block from the file's start up to the first "Result for" line is treated as iteration 1.
    Every time a line starting with "Result for" is encountered:
      - That line is added to the current block,
      - The block is closed and assigned the next iteration number (starting at 1),
      - Then a new block is started.
    If the file ends without a terminating "Result for" line, the remaining lines are stored with iteration as None.
    Returns a list of (iteration, block_lines) tuples.
    """
    blocks = []
    current_block_lines = []
    iteration_counter = 0

    for line in log_lines:
        if line.startswith("Result for"):
            current_block_lines.append(line)  # include the ending marker
            iteration_counter += 1
            blocks.append((iteration_counter, current_block_lines))
            current_block_lines = []  # reset block for next iteration
        else:
            current_block_lines.append(line)
    
    # If there is a leftover block without termination, add it with iteration None.
    if current_block_lines:
        blocks.append((None, current_block_lines))
    
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
            with open(args.logfile, "r", encoding="utf-8", errors="ignore") as f:
                log_lines = f.readlines()
        except UnicodeDecodeError:
            # 'utf-8' 실패 시 'latin-1'로 시도
            with open(args.logfile, "r", encoding="latin-1") as f:
                log_lines = f.readlines()
    except Exception as e:
        print(f"Could not open the log file: {e}")
        return

    blocks = parse_blocks(log_lines)

    print("=== Collision Count Summary ===")
    iterations = []
    collision_counts = []
    for iteration, block_lines in blocks:
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