"""
Example script (exp1.py) that sets up a Flow experiment configuration
and then calls train.py's train_rllib() to run the RL training.

Usage:
    python exp1.py
"""

import sys
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


class Exp1Submodule:
    """
    Submodule class containing Flow experiment configuration and training settings.
    This class is used by train.py's train_rllib() to retrieve flow_params and other settings.
    """
    # Experiment tag (must match the expected exp_config name)
    exp_tag = "figure8_with_rl"

    # Define Flow parameters
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


def main():
    """
    Main function that sets up the experiment configuration and calls train_rllib()
    to run RL training using the PPO algorithm via RLlib.
    """
    # (옵션) 만약 command-line 인자를 사용하고 싶다면, parse_args를 활용할 수 있음.
    # 여기서는 간단하게 Flags 클래스로 필요한 인자를 정의합니다.
    class Flags:
        exp_config = Exp1Submodule.exp_tag      # Experiment configuration name (must match exp_tag)
        rl_trainer = "rllib"                    # Trainer to use (here: "rllib")
        num_cpus = Exp1Submodule.N_CPUS         # Number of CPUs to use
        num_steps = 10                          # Total number of training iterations
        rollout_size = Exp1Submodule.N_ROLLOUTS # Number of rollouts per training iteration
        checkpoint_path = None                  # Optional: path to restore training from

    flags = Flags()

    # Call train_rllib() from train.py with the submodule and flags.
    train_rllib(Exp1Submodule, flags)


if __name__ == "__main__":
    main()
