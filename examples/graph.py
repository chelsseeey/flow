#!/usr/bin/env python3
import matplotlib.pyplot as plt

def main():
    # iteration별 collision 데이터 (Iteration 1부터 100까지)
    iterations = list(range(1, 101))
    collisions = [
        0, 0, 0, 2, 9, 17, 27, 26, 29, 26,
        22, 34, 28, 18, 19, 17, 26, 24, 26, 23,
        27, 15, 20, 16, 18, 12, 21, 21, 29, 25,
        32, 22, 23, 29, 23, 26, 28, 20, 21, 27,
        24, 26, 23, 31, 21, 15, 10, 7, 7, 5,
        2, 5, 15, 10, 4, 13, 7, 13, 9, 9,
        6, 4, 3, 8, 12, 4, 4, 8, 5, 11,
        5, 3, 5, 4, 5, 7, 5, 7, 15, 10,
        28, 21, 21, 16, 14, 25, 24, 19, 25, 23,
        10, 18, 14, 21, 8, 9, 9, 20, 7, 0
    ]

    # 그래프 설정
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, collisions, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Collision Count')
    plt.title('Collisions per Iteration')
    # x축 tick 간격을 5 단위로 설정 (과밀해질 경우 조정 가능)
    plt.xticks(range(0, 101, 5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
