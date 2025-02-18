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
    current_iter = None
    current_block = []

    for i, line in enumerate(log_lines):
        # 테이블 시작선 탐색
        if "+" in line and "-" in line and "|" in line:
            # 테이블 바로 아래(혹은 아래 2-3줄)에서 iteration 번호 추출
            # 예: "| PPO_MyCustomMergePOEnv-v2_609b9_00000 | RUNNING  | 203.255.177.230:18593 |     99 | ..."
            if i + 2 < len(log_lines):
                table_line = log_lines[i + 2]
                if "|" in table_line and "iter" in table_line:
                    try:
                        iter_num = int(table_line.split("|")[4].strip())
                    except (ValueError, IndexError):
                        iter_num = None
                    
                    # 이전 iteration이 있다면 저장
                    if current_iter is not None:
                        blocks.append((current_iter, current_block))
                    
                    # 새 iteration 시작
                    current_iter = iter_num
                    current_block = []
        
        # 충돌 로그 수집: 현재 iteration이 정해진 뒤에만 수집
        elif "Collision detected at time step" in line:
            if current_iter is not None:
                current_block.append(line)

    # 마지막 iteration 블록 추가
    if current_iter is not None and current_block:
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