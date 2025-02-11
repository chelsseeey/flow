import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

class CollisionLogger:
    def __init__(self):
        self.collision_count = 0
        self.iteration_collisions = []
        
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
        plot_collisions(self.iteration_collisions)
        return self.collision_count
        
    def __getstate__(self):
        return {
            'collision_count': self.collision_count,
            'iteration_collisions': self.iteration_collisions
        }
        
    def __setstate__(self, state):
        self.collision_count = state['collision_count']
        self.iteration_collisions = state['iteration_collisions']

def plot_collisions(collision_data):
    plt.figure()
    plt.plot(range(1, len(collision_data) + 1), collision_data, 'g-')
    plt.xlabel('Iteration Number')
    plt.ylabel('Number of Collisions')
    plt.title('Collisions per Iteration')
    plt.grid(True)
    plt.pause(0.01)

collision_logger = CollisionLogger()