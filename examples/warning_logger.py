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
        self.results_dir = "/root/ray_results"
        
    def __call__(self, env, worker):
        if not self.trial_name:
            # Get trial name from worker
            self.trial_name = worker.trial_name if hasattr(worker, 'trial_name') else None
            
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
        self.save_collision_data()
        return self.collision_count

    def save_collision_data(self):
        """Save collision data with trial name"""
        if self.trial_name:
            data = self.to_dict()
            save_path = os.path.join(self.results_dir, f"{self.trial_name}_collisions.json")
            os.makedirs(self.results_dir, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(data, f)

    def plot_collisions(self, trial_name=None):
        """Plot collision data for specific trial"""
        if trial_name:
            self.load_collision_data(trial_name)
        
        if self.trial_name:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(self.iteration_collisions) + 1), 
                    self.iteration_collisions, 'g-')
            plt.xlabel('Iteration Number')
            plt.ylabel('Number of Collisions')
            plt.title(f'Collisions - {self.trial_name}')
            plt.grid(True)
            plt.show()

    def load_collision_data(self, trial_name):
        """Load collision data for specific trial"""
        load_path = os.path.join(self.results_dir, f"{trial_name}_collisions.json")
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                data = json.load(f)
                self.from_dict(data)
                return True
        return False

    def to_dict(self):
        return {
            'trial_name': self.trial_name,
            'collision_count': self.collision_count,
            'iteration_collisions': self.iteration_collisions
        }

    def from_dict(self, data):
        self.trial_name = data['trial_name']
        self.collision_count = data['collision_count']
        self.iteration_collisions = data['iteration_collisions']

collision_logger = CollisionLogger()