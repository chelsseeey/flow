import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

class CollisionLogger:
    def __init__(self):
        # Data storage
        self.collision_count = 0
        self.iteration_collisions = []
        
        # Plotting setup (will cause circular references if serialized)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Real-time Collision Tracking')
        self.ax.grid(True)
        print("[DEBUG] CollisionLogger initialized")
        
    def __call__(self, env, worker):
        try:
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
                
            # Store count and update plot
            self.iteration_collisions.append(self.collision_count)
            self.update_plot()
            
            # Print current iteration summary
            print(f"Iteration {len(self.iteration_collisions)}: {self.collision_count} collisions")
            return self.collision_count
            
        except Exception as e:
            print(f"[ERROR] Error in __call__: {e}")
            return 0
        
    def update_plot(self):
        try:
            self.ax.clear()
            self.ax.plot(range(len(self.iteration_collisions)), 
                         self.iteration_collisions, 'r-', marker='o')
            self.ax.set_xlabel('Iteration')
            self.ax.set_ylabel('Number of Collisions')
            self.ax.set_title('Real-time Collision Tracking')
            self.ax.grid(True)
            plt.pause(0.01)
        except Exception as e:
            print(f"[ERROR] Plot update failed: {e}")

    def __getstate__(self):
        """
        Prevent circular references during serialization.
        Remove figure/axes from state so they won't be included in JSON/pickle.
        """
        state = self.__dict__.copy()
        # Remove matplotlib objects from serialization
        if 'fig' in state:
            del state['fig']
        if 'ax' in state:
            del state['ax']
        return state

    def __setstate__(self, state):
        """
        Restore state and recreate figure/axes after deserialization.
        """
        self.__dict__.update(state)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Real-time Collision Tracking')
        self.ax.grid(True)

collision_logger = CollisionLogger()