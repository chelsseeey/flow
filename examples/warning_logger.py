import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import json
import os

class CollisionLogger:
    def __init__(self):
        # Data storage
        self.collision_count = 0
        self.iteration_collisions = []
        self.trial_name = None
        self.results_dir = "/root/ray_results/figure8_with_lights"
        
        # Plotting setup - kept separate from data
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self._setup_plot()
        print("[DEBUG] CollisionLogger initialized")
        
    def _setup_plot(self):
        """Setup plot parameters"""
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Real-time Collision Tracking')
        self.ax.grid(True)
        
    def __call__(self, env, worker):
        if not self.trial_name:
            self.trial_name = worker.trial_name if hasattr(worker, 'trial_name') else None
            print(f"[DEBUG] Got trial name: {self.trial_name}")
            
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
            print(f"[DEBUG] Collision detected! Count: {self.collision_count}")
            
        # Store only the count
        self.iteration_collisions.append(self.collision_count)
        self.update_plot()
        return self.collision_count
        
    def update_plot(self):
        """Update real-time plot"""
        try:
            self.ax.clear()
            self.ax.plot(range(len(self.iteration_collisions)), 
                        self.iteration_collisions, 'r-', marker='o')
            self._setup_plot()
            if self.trial_name:
                self.ax.set_title(f'Real-time Collision Tracking - {self.trial_name}')
            plt.pause(0.01)
        except Exception as e:
            print(f"[ERROR] Plot update failed: {e}")

collision_logger = CollisionLogger()