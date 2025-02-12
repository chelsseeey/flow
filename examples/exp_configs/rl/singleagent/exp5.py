"""Figure eight example with traffic lights."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, TrafficLightParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import MultiAgentAccelPOEnv  # 변경
from flow.networks import FigureEightNetwork
from flow.utils.registry import make_create_env
from flow.core import rewards
import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from contextlib import redirect_stdout
plt.ion()  # Enable interactive mode

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 1

# Define the vehicles in the network.
# Add the RL-controlled vehicle (using the RLController and StaticLaneChanger)
vehicles = VehicleParams()
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,  # using the original value (you can change this if needed)
        decel=2.5,
    ),
    num_vehicles=1
)

# Add idm vehicles
vehicles.add(
    veh_id='idm',
    acceleration_controller=(IDMController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=7,
        decel=2.5,
    ),
    initial_speed=0,
    num_vehicles=13
)

# Define traffic light settings.
traffic_lights = TrafficLightParams(baseline=False)
traffic_lights.add(
    node_id="center",   # Node at the intersection in the figure eight
    tls_type="static",  # Static (fixed cycle) traffic light
    programID="1",
    phases=[
        {"duration": "5", "state": "GrGr"},  # Horizontal green
        {"duration": "3", "state": "yrGr"},   # Horizontal yellow
        {"duration": "2", "state": "rrrr"},   # All red
        {"duration": "5", "state": "rGrG"},    # Vertical green
        {"duration": "3", "state": "ryrG"},     # Vertical yellow
        {"duration": "2", "state": "rrrr"}      # All red
    ]
)

# Define the main flow parameters dictionary.
flow_params = dict(
    # Name of the experiment.
    exp_tag='figure8_with_lights',

    # The environment class to be used.
    env_name=MultiAgentAccelPOEnv,

    # The network class to be used.
    network=FigureEightNetwork,

    # The simulator to be used.
    simulator='traci',

    # SUMO-related parameters.
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        print_warnings=False,  # Disable debug prints
        restart_instance=True, # Restart
        emission_path=None,
        no_step_log=True,   # Disable step logs
    ),

    # Environment parameters.
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': 20,
            'max_accel': 3,
            'max_decel': 3,
            'sort_vehicles': False
        },
    ),

    # Network parameters.
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # Vehicles in the network.
    veh=vehicles,

    # Initial configuration parameters.
    initial=InitialConfig(),

    # Include the traffic light settings.
    tls=traffic_lights
)

# Create and register env
env = MultiAgentAccelPOEnv(
    env_params=flow_params['env'],
    sim_params=flow_params['sim'],
    network=FigureEightNetwork(
        name='figure_eight',
        vehicles=vehicles,
        net_params=flow_params['net'],
        initial_config=flow_params['initial'],
        traffic_lights=flow_params['tls']
    )
)


# Print each iteration's average reward
print("\nAll Iteration Average Rewards:")
for i, avg_reward in enumerate(all_iteration_rewards):
    print(f"Iteration {i}: {avg_reward}")


# Add RLlib configuration
from flow.examples.callbacks.collision_logger_callback import CollisionLoggerCallbacks
from ray import tune

config = {
    "env": MultiAgentAccelPOEnv,
    "env_config": flow_params,
    "num_workers": N_CPUS - 1,
    "rollout_fragment_length": HORIZON // N_ROLLOUTS,
    "callbacks": CollisionLoggerCallbacks,
}

if __name__ == "__main__":
    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 10},
        local_dir="/home/mcnl/Desktop/chaeyoung/flow/results",
    )