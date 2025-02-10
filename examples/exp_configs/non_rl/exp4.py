"""Figure eight example with traffic lights."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, TrafficLightParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import MultiAgentAccelPOEnv  # 변경
from flow.networks import FigureEightNetwork

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 1

# Define the vehicles in the network.
# Here we use one RL-controlled vehicle and 13 human-driven vehicles.
vehicles = VehicleParams()

# Add the RL-controlled vehicle (using the RLController and StaticLaneChanger)

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
    num_vehicles=14
)

# Define traffic light settings.
traffic_lights = TrafficLightParams(baseline=False)
traffic_lights.add(
    node_id="center",   # Node at the intersection in the figure eight
    tls_type="static",  # Static (fixed cycle) traffic light
    programID="1",
    phases=[
        {"duration": "10", "state": "GrGr"},  # Horizontal green
        {"duration": "3", "state": "yrGr"},   # Horizontal yellow
        {"duration": "2", "state": "rrrr"},   # All red
        {"duration": "10", "state": "rGrG"},    # Vertical green
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

print("Traffic light parameters:", flow_params['tls'])
print("Flow parameters:", flow_params)
