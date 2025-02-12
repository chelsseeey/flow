"""Figure eight example with traffic lights."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, TrafficLightParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import MultiAgentAccelPOEnv
from flow.networks import FigureEightNetwork
from flow.utils.registry import make_create_env
from flow.examples.warning_logger import collision_logger

# time horizon of a single rollout
HORIZON = 1500
N_ROLLOUTS = 20
N_CPUS = 1

vehicles = VehicleParams()
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,
        decel=2.5,
    ),
    num_vehicles=1
)
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

traffic_lights = TrafficLightParams(baseline=False)
traffic_lights.add(
    node_id="center",
    tls_type="static",
    programID="1",
    phases=[
        {"duration": "5", "state": "GrGr"},
        {"duration": "3", "state": "yrGr"},
        {"duration": "2", "state": "rrrr"},
        {"duration": "5", "state": "rGrG"},
        {"duration": "3", "state": "ryrG"},
        {"duration": "2", "state": "rrrr"}
    ]
)

flow_params = dict(
    exp_tag='figure8_with_lights',
    env_name=MultiAgentAccelPOEnv,
    network=FigureEightNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
    ),
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': 20,
            'max_accel': 3,
            'max_decel': 3,
            'sort_vehicles': False
        },
    ),
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),
    veh=vehicles,
    initial=InitialConfig(),
    tls=traffic_lights,
    callback=collision_logger,
)

def run_simulation(num_steps):
    # Create the environment
    create_env, _ = make_create_env(flow_params, version=0)
    env = create_env()

    # Run the simulation for a number of iterations
    for i in range(num_steps):
        print(f"\n[Iteration {i+1}]")
        # Call the collision logger
        collision_logger(env)
        
    print("Simulation done.")

if __name__ == "__main__":
    import sys
    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_simulation(num_steps)