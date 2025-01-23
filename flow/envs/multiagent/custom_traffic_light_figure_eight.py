# flow/envs/custom/custom_traffic_light_figure_eight.py

from flow.envs.base import Env
from flow.core import rewards
from gym import spaces
import numpy as np

class TrafficLightFigureEightEnv(Env):
    """Custom environment for Figure Eight Network with static traffic lights."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the environment."""
        super().__init__(env_params, sim_params, network, simulator)
        
        # Define a dummy action space since traffic lights are static
        self.action_space = spaces.Discrete(1)  # Only one possible action (no-op)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(self.env_params.additional_params["num_observed"],), 
            dtype=np.float32
        )

        # Initialize simulator
        self.sim = self.k

        # Initialize traffic light state if necessary
        self.current_phase = 0

    def reset(self):
        """Reset the environment to initial state."""
        # Reset the simulation
        super().reset()

        # Reset traffic lights
        self.reset_traffic_lights()

        # Return initial observation
        return self.get_state()

    def step(self, action):
        """Run one timestep of the environment's dynamics."""
        # Since action is a dummy, ignore it or treat as no-op
        # Apply action (no operation)
        # self.apply_action(action)

        # Step the simulation
        self.sim.step()

        # Get observation
        obs = self.get_state()

        # Compute reward
        reward = self.compute_reward(obs)

        # Check if done
        done = self.sim.get_time() >= self.env_params.horizon

        # Info dictionary can be used for debugging
        info = {}

        return obs, reward, done, info

    def get_state(self):
        """Get the current state of the environment."""
        # Collect speeds of observed vehicles
        observed_ids = self.k.vehicle.get_ids()
        speeds = [
            self.k.vehicle.get_speed(veh_id)
            for veh_id in observed_ids[:self.env_params.additional_params["num_observed"]]
        ]
        return np.array(speeds, dtype=np.float32)

    def compute_reward(self, obs):
        """Compute the reward for the current state."""
        # Example: Minimize total speed to reduce congestion
        return -np.sum(obs)
    
    def reset_traffic_lights(self):
        """Reset traffic lights to initial phase."""
        # 올바른 방식으로 TrafficLightParams 접근
        # 'tls'는 TrafficLightParams 객체의 리스트를 나타냄
        initial_phase = self.network.traffic_lights.tls[0]['phases'][0]['state']
        self.k.traffic_light.set_state("center0", initial_phase)
        self.current_phase = 0

    def _apply_rl_actions(self, actions):
        """No-op since traffic lights are static."""
        pass
