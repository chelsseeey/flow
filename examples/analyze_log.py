import argparse
import re
import matplotlib.pyplot as plt

def parse_blocks(log_lines):
    """
    Parses log file lines, extracting each iteration block.
    Each block ends with a line starting with "== Status ==".
    The block from the file's beginning up to the first "== Status ==" is the first block.
    Within each block, the iteration number is extracted by searching for the first occurrence of "iter".
    Returns a list of (iteration, block_lines) tuples for each block.
    If the iteration number is not found, it is marked as None.
    """
    blocks = []
    current_block_lines = []
    current_iteration = None

    for line in log_lines:
        # End of the current block is identified by a line starting with "== Status =="
        if line.startswith("== Status =="):
            if current_block_lines:
                blocks.append((current_iteration, current_block_lines))
            # Start a new block after the end marker
            current_block_lines = []
            current_iteration = None
        else:
            # Append the current line to the block
            current_block_lines.append(line)
            # Try to extract the iteration number if not already set.
            if current_iteration is None and "training_iteration:" in line:
                m = re.search(r"\|\s*(\d+)\s*\|", line)
                if m:
                    current_iteration = int(m.group(1))
    # Add the final block if any lines remain
    if current_block_lines:
        blocks.append((current_iteration, current_block_lines))
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

    # 그래프 그리기
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