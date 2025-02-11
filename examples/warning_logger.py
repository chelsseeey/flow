import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

class CollisionLogger:
    def __init__(self):
        self.collision_count = 0
        self.iteration_collisions = []
        # 그래프 초기화
        plt.ion()  # interactive mode 켜기
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.setup_plot()
        
    def setup_plot(self):
        self.ax.set_xlabel('Iteration Number')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Collisions per Iteration')
        self.ax.grid(True)
        
    def __call__(self, env, worker):
        f = io.StringIO()
        with redirect_stdout(f):
            state = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                
        output = f.getvalue()
        if "Collision detected at time step" in output:
            self.collision_count += 1
            print(f"Collision detected! Count: {self.collision_count}")
            
        self.iteration_collisions.append(self.collision_count)
        self.update_plot()  # 실시간 그래프 업데이트
        return self.collision_count
        
    def update_plot(self):
        self.ax.clear()
        self.setup_plot()
        self.ax.plot(range(1, len(self.iteration_collisions) + 1), 
                    self.iteration_collisions, 'g-')
        plt.pause(0.01)  # 그래프 업데이트
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # matplotlib 객체 제외
        del state['fig']
        del state['ax']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # matplotlib 객체 재생성
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.setup_plot()

collision_logger = CollisionLogger()