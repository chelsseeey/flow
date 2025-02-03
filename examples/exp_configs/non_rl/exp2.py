#!/usr/bin/env python
"""
Example script (exp2.py) that sets up a Flow experiment configuration
and then calls train.py's train_rllib() to run the RL training with a for-loop
that executes a specified number of training iterations and prints each iteration's result.

Usage:
    python exp2.py
"""

import os
import sys
import gym
from flow.core.params import (SumoParams, EnvParams, NetParams, VehicleParams,
                              SumoCarFollowingParams, TrafficLightParams)
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, RLController
from flow.envs.multiagent import MultiAgentAccelPOEnv
from flow.networks.figure_eight import FigureEightNetwork, ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env

# Import train.py's functions
from train import train_rllib, parse_args

print("[DEBUG] exp2.py is running.")

# ---------------------------
# Module-level experiment configuration (only parameter definitions)
# ---------------------------

exp_tag = "figure8_with_rl"

# Flow 환경 파라미터 (설정만; 실제 환경 생성 및 등록은 main() 내부에서 진행)
flow_params = dict(
    exp_tag=exp_tag,
    env_name=MultiAgentAccelPOEnv,
    network=FigureEightNetwork,
    simulator='traci',
    sim=SumoParams(render=False, sim_step=0.1),
    env=EnvParams(
        horizon=1500,  # 한 에피소드 당 최대 스텝 수
        additional_params={
            "target_velocity": 20,
            "max_accel": 3,
            "max_decel": 1.5,
            "sort_vehicles": False
        },
    ),
    net=NetParams(additional_params=ADDITIONAL_NET_PARAMS.copy()),
    veh=VehicleParams(),
    tls=TrafficLightParams(baseline=False)
)

N_CPUS = 1
N_ROLLOUTS = 20

# 차량 및 신호등 설정 (여기서는 파라미터만 정의)
flow_params["veh"].add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,
        accel=3,
        decel=1.5,
    ),
    num_vehicles=1
)

flow_params["veh"].add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=0,
        decel=2.5,
    ),
    initial_speed=0,
    num_vehicles=13
)

flow_params["tls"].add(
    node_id="center",
    tls_type="static",
    programID="1",
    phases=[
        {"duration": "10", "state": "GrGr"},
        {"duration": "3", "state": "yrGr"},
        {"duration": "2", "state": "rrrr"},
        {"duration": "10", "state": "rGrG"},
        {"duration": "3", "state": "ryrG"},
        {"duration": "2", "state": "rrrr"}
    ]
)

# RL 관련 추가 구성
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
# Define a simple Flags class for training arguments (설정만)
# ---------------------------
class Flags:
    exp_config = exp_tag
    rl_trainer = "rllib"
    num_cpus = N_CPUS
    num_steps = 10                # 총 학습 iteration 수 (예: 10)
    rollout_size = N_ROLLOUTS
    checkpoint_path = None
    ray_memory = 200 * 1024 * 1024

flags = Flags()

# ---------------------------
# Create a simple namespace object to hold experiment settings (설정만)
# ---------------------------
ExpConfig = type("ExpConfig", (), {})()
ExpConfig.flow_params = flow_params
ExpConfig.N_CPUS = N_CPUS
ExpConfig.N_ROLLOUTS = N_ROLLOUTS
ExpConfig.exp_tag = exp_tag

# ---------------------------
# Main function: environment initialization and training execution
# ---------------------------
def main():
    print("[DEBUG] Entering main() in exp2.py.")

    # 환경 등록 및 초기화는 main() 내부에서 수행합니다.
    from flow.utils.registry import register_env, make_create_env
    env_id = "MultiAgentAccelPOEnv-v0"
    if env_id not in gym.envs.registry.env_specs:
        create_env, env_name = make_create_env(params=flow_params, version=0)
        register_env(env_name, create_env)
        print("[DEBUG] Environment registered:", env_name)
    else:
        print("[DEBUG] Environment already registered.")
    
    print("[DEBUG] Starting training via train_rllib()")
    train_rllib(ExpConfig, flags)

if __name__ == "__main__":
    main()
