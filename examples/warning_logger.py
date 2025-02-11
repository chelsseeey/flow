import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

# 전역 플롯 객체
fig = None
ax = None

def setup_global_plot():
    global fig, ax
    if fig is None or ax is None:
        # 인터랙티브 모드
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Collisions')
        ax.set_title('Real-time Collision Tracking')
        ax.grid(True)
        # 그래프 창을 뜨게 하는 호출
        plt.show(block=False)

def update_global_plot(iterations):
    global fig, ax
    setup_global_plot()
    ax.clear()
    ax.plot(range(len(iterations)), iterations, 'r-', marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Collisions')
    ax.set_title('Real-time Collision Tracking')
    ax.grid(True)
    # 그래프 새로고침
    plt.draw()
    plt.pause(0.01)

class CollisionLogger:
    def __init__(self):
        self.collision_count = 0
        self.iteration_collisions = []
        print("[DEBUG] CollisionLogger initialized")

    def __call__(self, env, worker):
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                state = env.reset()
                done = False
                while not done:
                    action = env.action_space.sample()
                    next_state, reward, done, info = env.step(action)
            output = f.getvalue()

            # 충돌 메시지 감지
            if "Collision detected at time step" in output:
                self.collision_count += 1
                print(f"[DEBUG] Collision detected! Count: {self.collision_count}")

            # 매 iteration마다 충돌 개수 기록 후 그래프 업데이트
            self.iteration_collisions.append(self.collision_count)
            update_global_plot(self.iteration_collisions)
            print(f"Iteration {len(self.iteration_collisions)}: {self.collision_count} collisions")

            return self.collision_count
        except Exception as e:
            print(f"[ERROR] Error in __call__: {e}")
            return 0

# 인스턴스 생성
collision_logger = CollisionLogger()

# 학습 시작할 때 한 번 실행해주면 그래프 창이 바로 뜸
setup_global_plot()
update_global_plot([0])