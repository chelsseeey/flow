import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

# 전역 플롯 객체
fig = None
ax = None

def setup_global_plot():
    global fig, ax
    if fig is None or ax is None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Collisions')
        ax.set_title('Real-time Collision Tracking')
        ax.grid(True)
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
    plt.draw()
    plt.pause(0.01)

class CollisionLogger:
    def __init__(self):
        self.collision_count = 0
        self.iteration_collisions = []
        print("[DEBUG] CollisionLogger initialized")

    def __call__(self, env, worker=None):
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                state = env.reset()
                done = False
                while not done:
                    action = env.action_space.sample()
                    next_state, reward, done, info = env.step(action)
            output = f.getvalue()

            if "Collision detected" in output or "Collision occurred" in output:
                self.collision_count += 1
                print(f"[DEBUG] Collision detected! Count: {self.collision_count}")

            self.iteration_collisions.append(self.collision_count)
            update_global_plot(self.iteration_collisions)
            print(f"Iteration {len(self.iteration_collisions)}: {self.collision_count} collisions")

            return self.collision_count
        except Exception as e:
            print(f"[ERROR] Error in CollisionLogger: {e}")
            return 0

collision_logger = CollisionLogger()

def monitor_collisions(env, worker=None):
    return collision_logger(env, worker)