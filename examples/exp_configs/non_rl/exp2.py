#!/usr/bin/env python
"""
Example script (exp2.py) that sets up a Flow experiment configuration
and then calls train.py's train_rllib() to run the RL training.
"""

import os
import sys

if __name__ == "__main__":
    # Add parent directory to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Flow imports
from flow.core.params import (
    SumoParams,
    EnvParams,
    NetParams,
    VehicleParams,
    SumoCarFollowingParams,
    TrafficLightParams
)
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, RLController
from flow.envs.multiagent import MultiAgentAccelPOEnv
from flow.networks.figure_eight import FigureEightNetwork, ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env

# Import train.py's functions
from train import train_rllib, parse_args

print("[DEBUG] exp2.py is running.")
# ---------------------------
# Module-level experiment configuration
# ---------------------------

# Experiment tag
exp_tag = "figure8_with_rl"

# Flow parameters
flow_params = dict(
    exp_tag=exp_tag,
    env_name=MultiAgentAccelPOEnv,
    network=FigureEightNetwork,
    simulator='traci',
    sim=SumoParams(
        render=False,      # Disable GUI rendering for faster training
        sim_step=0.1,      # Simulation time step in seconds
    ),
    env=EnvParams(
        horizon=1500,  # Number of steps per rollout
        additional_params={
            "target_velocity": 20,   # Target velocity (m/s)
            "max_accel": 3,          # Maximum acceleration (m/s²)
            "max_decel": 1.5,        # Maximum deceleration (m/s²)
            "sort_vehicles": False   # Do not sort vehicles by ID
        },
    ),
    net=NetParams(additional_params=ADDITIONAL_NET_PARAMS.copy()),
    veh=VehicleParams(),
    tls=TrafficLightParams(baseline=False)
)

# Training parameters
N_CPUS = 1         # Number of CPUs (workers) to use
N_ROLLOUTS = 20    # Number of rollouts per training iteration

# --- Define Vehicles ---
# Add RL-controlled vehicle
flow_params["veh"].add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,  # SUMO speed mode (allows some collisions)
        accel=3,
        decel=1.5,
    ),
    num_vehicles=1
)

# Add IDM-controlled vehicles
flow_params["veh"].add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=0,   # Default speed mode (prevents collisions)
        decel=2.5,
    ),
    initial_speed=0,
    num_vehicles=13
)

# --- Define Traffic Lights ---
flow_params["tls"].add(
    node_id="center",
    tls_type="static",
    programID="1",
    phases=[
        {"duration": "10", "state": "GrGr"},
        {"duration": "3",  "state": "yrGr"},
        {"duration": "2",  "state": "rrrr"},
        {"duration": "10", "state": "rGrG"},
        {"duration": "3",  "state": "ryrG"},
        {"duration": "2",  "state": "rrrr"}
    ]
)

# --- Add RL configuration ---
flow_params.update({
    "algorithm": "PPO",
    "model": {"fcnet_hiddens": [32, 32, 32]},
    "gamma": 0.999,
    "lambda": 0.97,
    "kl_target": 0.02,
    "num_sgd_iter": 10,
    "multiagent": {
        "policies": None,
        "policy_mapping_fn": None,
        "policies_to_train": None
    }
})

# ---------------------------
# Define a simple Flags class for training arguments
# ---------------------------
class Flags:
    exp_config = exp_tag          # Name of the experiment configuration
    rl_trainer = "rllib"          # Trainer to use (here: "rllib")
    num_cpus = N_CPUS             # Number of CPUs to use
    num_steps = 10                # Total number of training iterations
    rollout_size = N_ROLLOUTS     # Number of rollouts per training iteration
    checkpoint_path = None        # Path to restore training from (if any)
    ray_memory = 200 * 1024 * 1024  # Ray 메모리 설정 추가

flags = Flags()

# ---------------------------
# Create a simple namespace object to hold experiment settings
# ---------------------------
# Since train.py's train_rllib() expects an object with attributes:
# flow_params, N_CPUS, N_ROLLOUTS, and (optionally) exp_tag,
# we create an empty object and attach these attributes.
ExpConfig = type("ExpConfig", (), {})()  # Create an empty object
ExpConfig.flow_params = flow_params
ExpConfig.N_CPUS = N_CPUS
ExpConfig.N_ROLLOUTS = N_ROLLOUTS
ExpConfig.exp_tag = exp_tag

# ---------------------------
# Main function: call train_rllib() from train.py
# ---------------------------
def main():
    # Optionally, you could also parse command-line arguments:
    # flags = parse_args(sys.argv[1:])
    print("[DEBUG] Entering main() in exp2.py.")
    train_rllib(ExpConfig, flags)


if __name__ == "__main__":
    main()


