import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

class CollisionLogger:
    def __init__(self):
        self.collision_count = 0
        self.iteration_collisions = []
        self._setup_plot()
        
    def _setup_plot(self):
        """Setup matplotlib plot - not included in serialization"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
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
        self._update_plot()
        return self.collision_count
        
    def _update_plot(self):
        """Update plot - not included in serialization"""
        if not hasattr(self, 'ax'):
            self._setup_plot()
        self.ax.clear()
        self.ax.plot(range(1, len(self.iteration_collisions) + 1), 
                    self.iteration_collisions, 'g-')
        self.ax.set_xlabel('Iteration Number')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Collisions per Iteration')
        self.ax.grid(True)
        plt.pause(0.01)
        
    def __getstate__(self):
        """Only serialize collision data"""
        return {
            'collision_count': self.collision_count,
            'iteration_collisions': self.iteration_collisions
        }
    
    def __setstate__(self, state):
        """Restore collision data and recreate plot"""
        self.collision_count = state['collision_count']
        self.iteration_collisions = state['iteration_collisions']
        self._setup_plot()

collision_logger = CollisionLogger()