import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from datetime import datetime
import os

class CollisionLogger:
    def __init__(self):
        # Setup plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_xlabel('Iteration Number')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Collisions per Iteration')
        self.ax.grid(True)
        
        # Collision tracking
        self.iteration_collisions = []
        
    def __call__(self, env, worker):
        collision_count = 0
        f = io.StringIO()
        with redirect_stdout(f):
            state = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                
        # Check for collision
        output = f.getvalue()
        if "Collision detected at time step" in output:
            collision_count += 1
            
        # Update data and plot
        self.iteration_collisions.append(collision_count)
        self._update_plot()
        
    def _update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('Iteration Number')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Collisions per Iteration')
        self.ax.grid(True)
        self.ax.plot(range(1, len(self.iteration_collisions) + 1), 
                    self.iteration_collisions, 'g-')
        plt.pause(0.01)

# Create global logger instance
collision_logger = CollisionLogger()