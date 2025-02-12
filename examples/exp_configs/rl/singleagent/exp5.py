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

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

# First two subplots same as before
ax1.set_xlabel('Rollout Number')
ax1.set_ylabel('Reward')
ax1.set_title('Rewards per Rollout')
ax1.grid(True)
ax1.set_ylim(100, 250)
ax1.set_yticks(range(100, 251))

ax2.set_xlabel('Iteration Number')
ax2.set_ylabel('Average Reward')
ax2.set_title('Average Rewards per Iteration')
ax2.grid(True)
ax2.set_ylim(100, 250)
ax2.set_yticks(range(100, 251))

# Add collision plot
ax3.set_xlabel('Iteration Number')
ax3.set_ylabel('Number of Collisions')
ax3.set_title('Collisions per Iteration')
ax3.grid(True)
ax3.set_ylim(0, 20)
ax3.set_yticks(range(0, 21, 5))

# Evaluation variables
num_iterations = 10
all_iteration_rewards = []
all_rollout_rewards = []
iteration_collisions = []  # New: track collisions per iteration

# Iteration loop
for i in range(1, num_iterations + 1):
    iteration_rewards = []
    collision_count = 0
    
    for r in range(N_ROLLOUTS):
        # Capture stdout to detect collision messages
        f = io.StringIO()
        with redirect_stdout(f):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 단일 에이전트 환경에서는 "rl" 키에 대해 액션 값을 dict로 전달합니다.
                action = {'rl_0': env.action_space.sample()}
                next_state, _, done, info = env.step(action)
                # 환경의 내장 보상 함수(compute_reward)를 호출합니다.
                reward_dict = env.compute_reward(action, fail=False)
                # 단일 에이전트이므로, 'rl_0' 키에 해당하는 보상을 사용합니다.
                reward = reward_dict['rl_0']
                episode_reward += reward
        
        # Check captured output for collision
        output = f.getvalue()
        if "Collision detected at time step" in output:
            collision_count += 1
            
        iteration_rewards.append(episode_reward)
        all_rollout_rewards.append(episode_reward)
        print(f"Rollout {r} in Iteration {i}: Reward = {episode_reward}")
        
        # Update rollout plot
        ax1.clear()
        ax1.set_xlabel('Rollout Number')
        ax1.set_ylabel('Reward')
        ax1.set_title('All Rollout Rewards')
        ax1.grid(True)
        ax1.set_ylim(20, 250)
        ax1.set_yticks(range(20, 251))
        ax1.plot(range(len(all_rollout_rewards)), all_rollout_rewards, 'b-')
        plt.pause(0.01)
    
    # Store iteration data and update plots
    curr_iter_avg = np.mean(iteration_rewards)
    all_iteration_rewards.append(curr_iter_avg)
    iteration_collisions.append(collision_count)
    print(f"\nIteration {i} Average Reward: {curr_iter_avg}")
    print(f"Iteration {i} Collisions: {collision_count}\n")
    
    # Update iteration plots
    ax2.clear()
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Average Rewards per Iteration')
    ax2.grid(True)
    ax2.set_ylim(20, 250)
    ax2.set_yticks(range(20, 251))
    ax2.plot(range(1, len(all_iteration_rewards) + 1), all_iteration_rewards, 'r-')
    
    ax3.clear()
    ax3.set_xlabel('Iteration Number')
    ax3.set_ylabel('Number of Collisions')
    ax3.set_title('Collisions per Iteration')
    ax3.grid(True)
    ax3.set_ylim(0, 30)
    ax3.set_yticks(range(0, 31, 5))
    ax3.plot(range(1, len(iteration_collisions) + 1), iteration_collisions, 'g-')
    plt.pause(0.01)

plt.show()

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