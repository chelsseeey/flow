# exp2.py

"""
Example script (exp1.py) that sets up a Flow experiment configuration
and then calls train.py's train_rllib() to run the RL training.
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

# Import train.py's train_rllib and parse_args functions
from train import train_rllib, parse_args

# ##############################################################################
# 1) Define the experiment configuration as a submodule
# ##############################################################################
class Exp1Submodule:
    """
    Submodule class containing Flow parameters and training settings for the experiment.
    """

    # Flow experiment tag
    exp_tag = "exp1_figure_eight_with_rl"

    # Define Flow parameters
    flow_params = dict(
        exp_tag=exp_tag,
        env_name=MultiAgentAccelPOEnv,
        network=FigureEightNetwork,
        simulator='traci',
        sim=SumoParams(
            render=False,      # Whether to render the GUI
            sim_step=0.1,      # Simulation step size in seconds
        ),
        env=EnvParams(
            horizon=1500,  # Number of steps per rollout
            additional_params={
                "target_velocity": 20,   # Target velocity for RL vehicles (m/s)
                "max_accel": 3,          # Max acceleration for RL vehicles (m/s²)
                "max_decel": 1.5,        # Max deceleration for RL vehicles (m/s²)
                "sort_vehicles": False   # Whether to sort vehicles by ID
            },
        ),
        net=NetParams(
            additional_params=ADDITIONAL_NET_PARAMS.copy()
        ),
        veh=VehicleParams(),
        tls=TrafficLightParams()
    )

    # Number of CPUs and rollouts
    N_CPUS = 1
    N_ROLLOUTS = 20

    # Define vehicles
    # Add RL controlled vehicle
    flow_params["veh"].add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        lane_change_controller=(StaticLaneChanger, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=31,  # SUMO speed mode (31 allows some collisions)
            accel=3,
            decel=1.5,
        ),
        num_vehicles=1
    )

    # Add IDM controlled vehicles
    flow_params["veh"].add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        lane_change_controller=(StaticLaneChanger, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=0,  # Default speed mode (prevents collisions)
            decel=2.5,
        ),
        initial_speed=0,
        num_vehicles=13
    )

    # Define Traffic Lights
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


# ##############################################################################
# 2) Define the main function to set up the experiment and call train_rllib
# ##############################################################################
def main():
    """Set up the experiment and call train_rllib() to perform RL training."""
    # Instantiate the experiment submodule
    submodule = Exp1Submodule()

    # Define a simple Flags class to hold training arguments
    class Flags:
        exp_config = submodule.exp_tag          # Name of the experiment configuration
        rl_trainer = "rllib"                    # Trainer to use
        num_cpus = submodule.N_CPUS             # Number of CPUs to use
        num_steps = 10                          # Number of training iterations
        rollout_size = submodule.N_ROLLOUTS     # Number of rollouts per training iteration
        checkpoint_path = None                  # Path to restore from, if any

    flags = Flags()

    # Call train_rllib with the submodule and flags
    train_rllib(submodule, flags)


if __name__ == "__main__":
    main()
