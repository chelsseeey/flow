import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import json

class CollisionLogger:
    def __init__(self):
        self.collision_count = 0
        self.iteration_collisions = []
        print("CollisionLogger initialized")
        
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
        return self.collision_count

    def to_dict(self):
        """Convert to JSON serializable format"""
        return {
            'collision_count': self.collision_count,
            'iteration_collisions': self.iteration_collisions
        }

    @classmethod
    def from_dict(cls, data):
        """Create instance from dict data"""
        instance = cls()
        instance.collision_count = data['collision_count']
        instance.iteration_collisions = data['iteration_collisions']
        return instance

    def plot_collisions(self):
        """Plot collision data"""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.iteration_collisions) + 1), 
                self.iteration_collisions, 'g-')
        plt.xlabel('Iteration Number')
        plt.ylabel('Number of Collisions')
        plt.title('Collisions per Iteration')
        plt.grid(True)
        plt.show()

collision_logger = CollisionLogger()