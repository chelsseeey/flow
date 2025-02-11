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
        self.results_dir = "/root/ray_results/figure8_with_lights"
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
        self.save_collision_data()
        return self.collision_count

    def save_collision_data(self):
        """Save collision data with trial name"""
        if self.trial_name:
            try:
                data = self.to_dict()
                save_path = os.path.join(self.results_dir, self.trial_name, "collision_data.json")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"[DEBUG] Saving collision data to: {save_path}")
                with open(save_path, 'w') as f:
                    json.dump(data, f)
                print("[DEBUG] Successfully saved collision data")
            except Exception as e:
                print(f"[ERROR] Failed to save collision data: {e}")

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
        load_path = os.path.join(self.results_dir, trial_name, "collision_data.json")
        print(f"[DEBUG] Looking for collision data at: {load_path}")
        if os.path.exists(load_path):
            print(f"[DEBUG] Found collision data file")
            try:
                with open(load_path, 'r') as f:
                    data = json.load(f)
                    self.from_dict(data)
                    return True
            except Exception as e:
                print(f"[ERROR] Failed to load collision data: {e}")
                return False
        print(f"[DEBUG] No collision data file found")
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