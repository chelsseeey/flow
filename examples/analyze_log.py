#!/usr/bin/env python3
import argparse
import re

def parse_blocks(log_lines):
    """
    로그 파일의 줄들을 순회하면서, 각 iteration 블록을 추출합니다.
    각 블록은 "Result for"로 시작하며, 그 블록 내에서 "training_iteration:"을 찾아 iteration 번호를 기록합니다.
    반환값은 각 블록을 (iteration, block_lines) 튜플의 리스트로 반환합니다.
    iteration 번호가 없으면 None으로 표시됩니다.
    """
    blocks = []
    current_block_lines = []
    current_iteration = None

    for line in log_lines:
        # 새로운 블록의 시작: "Result for"로 시작하는 줄
        if line.startswith("Result for"):
            # 이미 진행 중인 블록이 있다면 저장
            if current_block_lines:
                blocks.append((current_iteration, current_block_lines))
            # 새 블록 초기화
            current_block_lines = [line]
            current_iteration = None
        else:
            # 현재 블록이 진행 중일 때만 기록
            if current_block_lines:
                current_block_lines.append(line)
                # 블록 내에 training_iteration 정보가 아직 없다면 찾기
                if current_iteration is None and "training_iteration:" in line:
                    m = re.search(r"training_iteration:\s*(\d+)", line)
                    if m:
                        current_iteration = int(m.group(1))
    # 마지막 블록 저장
    if current_block_lines:
        blocks.append((current_iteration, current_block_lines))
    return blocks

def count_collisions_in_block(block_lines):
    return sum(1 for line in block_lines if "Collision detected at time step" in line)

def main():
    parser = argparse.ArgumentParser(
        description="로그 파일에서 각 iteration 별로 'Collision detected at time step' 발생 횟수 계산."
    )
    parser.add_argument("logfile", help="로그 파일의 경로")
    args = parser.parse_args()

    try:
        with open(args.logfile, "r") as f:
            log_lines = f.readlines()
    except Exception as e:
        print(f"로그 파일을 열 수 없습니다: {e}")
        return

    blocks = parse_blocks(log_lines)
    if not blocks:
        print("로그 파일에서 iteration 블록을 찾을 수 없습니다.")
        return

    print("=== 충돌 횟수 요약 ===")
    for iteration, block_lines in blocks:
        # iteration 번호가 없으면 '알 수 없음'으로 출력
        iteration_label = str(iteration) if iteration is not None else "알 수 없음"
        collision_count = count_collisions_in_block(block_lines)
        print(f"Iteration {iteration_label}: {collision_count} collisions")

if __name__ == "__main__":
    main()
