#!/usr/bin/env python3
import argparse
import re

def parse_log_file(log_lines):
    """
    로그 파일의 줄 목록을 받아서, 각 iteration 블록의 시작 인덱스와
    training_iteration 번호를 리스트로 반환합니다.
    
    반환 예:
        [(line_index1, iteration1), (line_index2, iteration2), ...]
    """
    blocks = []
    # 로그 블록의 시작을 "Result for" 라인으로 판단하고,
    # 이후 몇 줄 안에서 "training_iteration:" 값을 찾습니다.
    for i, line in enumerate(log_lines):
        if line.startswith("Result for"):
            iteration = None
            # 이후 20줄 정도 안에서 training_iteration을 찾습니다.
            for j in range(i, min(i + 20, len(log_lines))):
                m = re.search(r"training_iteration:\s*(\d+)", log_lines[j])
                if m:
                    iteration = int(m.group(1))
                    break
            if iteration is not None:
                blocks.append((i, iteration))
    return blocks

def count_collisions_per_iteration(log_lines, blocks):
    """
    blocks에 기록된 각 블록을 기준으로 구간을 나눈 후, 각 구간 내에서
    "Collision detected at time step" 문구의 등장 횟수를 센다.
    
    만약 블록이 하나도 없다면 전체 파일에 대해 충돌 횟수를 센다.
    """
    collision_counts = {}
    if not blocks:
        # iteration 블록이 없으면 전체 파일의 충돌 횟수를 계산
        total = sum(1 for line in log_lines if "Collision detected at time step" in line)
        collision_counts["전체"] = total
        return collision_counts

    # 첫 블록 전의 줄도 첫 iteration에 포함(블록이 시작되기 전의 로그도 해당 iteration의 일부로 간주)
    for idx, (start_idx, iteration) in enumerate(blocks):
        if idx == 0:
            segment = log_lines[0: (blocks[idx + 1][0] if len(blocks) > 1 else len(log_lines))]
        else:
            # 현재 블록 시작부터 다음 블록 시작 전까지
            end_idx = blocks[idx + 1][0] if idx < len(blocks) - 1 else len(log_lines)
            segment = log_lines[start_idx:end_idx]
        count = sum(1 for line in segment if "Collision detected at time step" in line)
        collision_counts[iteration] = count
    return collision_counts

def main():
    parser = argparse.ArgumentParser(
        description="각 iteration별로 'Collision detected at time step' 발생 횟수를 계산합니다."
    )
    parser.add_argument("logfile", help="로그 파일의 경로")
    args = parser.parse_args()

    try:
        with open(args.logfile, "r") as f:
            log_lines = f.readlines()
    except Exception as e:
        print(f"로그 파일을 여는 중 오류 발생: {e}")
        return

    blocks = parse_log_file(log_lines)
    counts = count_collisions_per_iteration(log_lines, blocks)

    print("\n=== 충돌 횟수 요약 ===")
    for iteration in sorted(counts, key=lambda x: (x if isinstance(x, int) else -1)):
        print(f"Iteration {iteration}: {counts[iteration]} collisions")

if __name__ == "__main__":
    main()
