import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

class CollisionLogger:
    def __init__(self):
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.setup_plot()
        self.collision_count = 0
        self.iteration_collisions = []
        
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
        
        self.iteration_collisions.append(self.collision_count)
        self.update_plot()
        plt.draw()  # Force update
        plt.pause(0.01)  # Allow GUI events
        return self.collision_count
        
    def update_plot(self):
        self.ax.clear()
        self.setup_plot()
        self.ax.plot(range(1, len(self.iteration_collisions) + 1), 
                    self.iteration_collisions, 'g-')
        self.fig.canvas.flush_events()  # Force update display
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove matplotlib objects before serialization
        del state['fig']
        del state['ax']
        return state
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.setup_plot()

collision_logger = CollisionLogger()