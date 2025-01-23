# custom_traffic_light_figure_eight.py

from flow.envs.base import Env
from flow.core import rewards
from gym import spaces

import numpy as np
from gym.spaces import Box

class TrafficLightFigureEightEnv(Env):
    """Custom environment for Figure Eight Network with traffic lights."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the environment."""
        super().__init__(env_params, sim_params, network, simulator)
        
        # Define action and observation space
        # Example: Continuous action space for traffic light phases
        self.action_space = spaces.Discrete(4)  # 4 possible phases
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.env_params.additional_params["num_observed"],), dtype=np.float32
        )

        # Initialize traffic light state if necessary
        # Example: Current phase index
        self.current_phase = 0

    def reset(self):
        """Reset the environment to initial state."""
        # Reset the simulation
        self.sim.reset()

        # Reset traffic lights
        self.reset_traffic_lights()

        # Return initial observation
        return self.get_state()

    def step(self, action):
        """Run one timestep of the environment's dynamics."""
        # Apply action (e.g., change traffic light phase)
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
        # Example: Collect speeds of observed vehicles
        observed_ids = self.k.vehicle.get_ids()
        speeds = [self.k.vehicle.get_speed(veh_id) for veh_id in observed_ids]
        return np.array(speeds)

    def compute_reward(self, obs):
        """Compute the reward for the current state."""
        # Example: Minimize total speed (to reduce congestion)
        return -np.sum(obs)
    
    def reset_traffic_lights(self):
        """Reset traffic lights to initial phase."""
        initial_phase = self.network.traffic_lights[0].phases[0]["state"]
        self.k.traffic_light.set_state("center0", initial_phase)
        self.current_phase = 0


    def action_space(self):
        """행동 공간 정의"""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles,),
            dtype=np.float32
        )

    def observation_space(self):
        """관찰 공간 정의"""
        return Box(
            low=0,
            high=1,
            shape=(2 * self.initial_vehicles.num_vehicles,),
            dtype=np.float32
        )