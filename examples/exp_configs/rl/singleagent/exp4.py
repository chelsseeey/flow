"""Figure eight example with traffic lights."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, TrafficLightParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs import Env
from flow.networks import FigureEightNetwork
from flow.utils.registry import make_create_env
from flow.core import rewards
import gym
import numpy as np

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 1

# Define the vehicles in the network.
vehicles = VehicleParams()

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
    env_name=Env,
    
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

# Create and register env
create_env, env_name = make_create_env(params=flow_params, version=0)
env = create_env(0)

# Evaluation with iterations and rollouts
num_iterations = 10
all_iteration_rewards = []  # 각 iteration의 평균 reward 저장

# Iteration loop
for i in range(num_iterations):
    iteration_rewards = []  # 현재 iteration의 rollout rewards
    
    # 20 rollouts for this iteration
    for r in range(N_ROLLOUTS):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            reward = rewards.desired_velocity(env, fail=False)
            episode_reward += reward
            
        iteration_rewards.append(episode_reward)
        print(f"Rollout {r} in Iteration {i}: Reward = {episode_reward}")
    
    # 현재 iteration의 평균 reward 계산
    curr_iter_avg = np.mean(iteration_rewards)
    all_iteration_rewards.append(curr_iter_avg)
    print(f"\nIteration {i} Average Reward: {curr_iter_avg}\n")

# Print each iteration's average reward
print("\nAll Iteration Average Rewards:")
for i, avg_reward in enumerate(all_iteration_rewards):
    print(f"Iteration {i}: {avg_reward}")

# Original prints
print("\nTraffic light parameters:", flow_params['tls'])
print("Flow parameters:", flow_params)