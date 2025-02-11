import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import json
import os

class CollisionLogger:
    def __init__(self):
        self.collision_count = 0
        self.iteration_collisions = []
        self.trial_name = None
        
        # Initialize real-time plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Real-time Collision Tracking')
        self.ax.grid(True)
        print("[DEBUG] CollisionLogger initialized")
        
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
            print(f"[DEBUG] Current trial: {self.trial_name}")
            
        self.iteration_collisions.append(self.collision_count)
        self.update_plot()  # Just update plot in real-time
        return self.collision_count
        
    def update_plot(self):
        """Update real-time plot"""
        self.ax.clear()
        self.ax.plot(range(len(self.iteration_collisions)), 
                    self.iteration_collisions, 'r-', marker='o')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title(f'Real-time Collision Tracking - {self.trial_name}')
        self.ax.grid(True)
        plt.pause(0.01)  # Small pause to update display

collision_logger = CollisionLogger()