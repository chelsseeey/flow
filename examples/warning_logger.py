import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class CollisionLogger:
    def __init__(self):
        self.iteration_collisions = []
        self._setup_plotting()
        self._setup_logging()
        
    def _setup_plotting(self):
        """Initialize matplotlib plot"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self._update_plot()
        
    def _setup_logging(self):
        """Setup logging directory and file"""
        self.log_dir = "collision_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = f"{self.log_dir}/collisions_{datetime.now():%Y%m%d_%H%M%S}.txt"
        
    def __call__(self, env, worker):
        """Track collisions in simulation"""
        collision_count = 0
        f = io.StringIO()
        with redirect_stdout(f):
            state = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                
        output = f.getvalue()
        if "Collision detected at time step" in output:
            collision_count += 1
            self._log_collision(collision_count)
            
        self.iteration_collisions.append(collision_count)
        self._update_plot()
        
    def _log_collision(self, count):
        """Log collision to file"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Collision count: {count}\n")
            
    def _update_plot(self):
        """Update matplotlib plot"""
        if not hasattr(self, 'ax'):
            self._setup_plotting()
        self.ax.clear()
        self.ax.set_xlabel('Iteration Number')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Collisions per Iteration')
        self.ax.grid(True)
        if self.iteration_collisions:
            self.ax.plot(range(1, len(self.iteration_collisions) + 1), 
                        self.iteration_collisions, 'g-')
        plt.pause(0.01)
        
    def __getstate__(self):
        """Make class serializable"""
        state = self.__dict__.copy()
        # Don't pickle matplotlib objects
        del state['fig']
        del state['ax']
        return state
    
    def __setstate__(self, state):
        """Restore from serialization"""
        self.__dict__.update(state)
        self._setup_plotting()

collision_logger = CollisionLogger()