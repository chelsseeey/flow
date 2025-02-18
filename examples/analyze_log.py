import argparse
import matplotlib
matplotlib.use('Agg')

def parse_blocks(log_lines):
    """Parse log lines into iteration blocks."""
    blocks = []
    current_block = []
    current_iter = None
    
    for line in log_lines:
        # Detect iteration start
        if "Iteration" in line:
            try:
                # Simple iteration number extraction
                iter_str = line.split("Iteration")[1].split(":")[0].strip()
                iter_num = int(iter_str)
                if current_iter is not None and current_block:
                    blocks.append((current_iter, current_block))
                current_iter = iter_num
                current_block = []
            except (ValueError, IndexError):
                continue
        
        # Add collision lines to current block
        if "Collision detected" in line:
            current_block.append(line)
    
    # Add the last block
    if current_iter is not None and current_block:
        blocks.append((current_iter, current_block))
    
    return blocks

def count_collisions_in_block(block_lines):
    """Count collisions in a block."""
    return len([line for line in block_lines if "Collision detected" in line])

def main():
    parser = argparse.ArgumentParser(
        description="Analyzes collision occurrences in training log file."
    )
    parser.add_argument("logfile", help="Path to the log file")
    args = parser.parse_args()

    try:
        with open(args.logfile, "r", encoding='utf-8', errors='ignore') as f:
            print("\nReading log file...")
            log_lines = f.readlines()
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    print("Parsing log lines...")
    blocks = parse_blocks(log_lines)
    
    if not blocks:
        print("No iteration blocks found in the log file.")
        return
    
    print("\n=== Collision Analysis Summary ===")
    for iteration, block_lines in sorted(blocks):
        collision_count = count_collisions_in_block(block_lines)
        if collision_count > 0:
            print(f"Iteration {iteration}: {collision_count} collisions")

if __name__ == "__main__":
    main()