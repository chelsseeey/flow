"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
on-ramp merge to a single lane open highway network.
"""

import numpy as np
from gym.spaces import Box
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import (SumoParams, EnvParams, InitialConfig,
                              NetParams, InFlows, SumoCarFollowingParams,
                              TrafficLightParams, VehicleParams)
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.controllers import IDMController, RLController
from flow.envs.multiagent import MultiAgentMergePOEnv
from flow.networks import MergeNetwork
from flow.utils.registry import make_create_env

# 사용자 정의 환경 클래스
class MyCustomMergePOEnv(MultiAgentMergePOEnv):
    """커스텀 관측 공간을 갖는 MultiAgentMergePOEnv."""
    def __init__(self, env_params, sim_params, network, simulator="traci"):
        super().__init__(env_params, sim_params, network, simulator)
        
        # 원하는 관측 범위를 여기서 재정의합니다
        self._custom_obs_space = Box(
            low=np.array([-1e8]*5, dtype=np.float32),
            high=np.array([1e8]*5, dtype=np.float32),
            dtype=np.float32
        )

    @property
    def observation_space(self):
        return self._custom_obs_space

    @observation_space.setter
    def observation_space(self, value):
        # 필요할 경우 override 가능
        self._custom_obs_space = value

# experiment number
# - 0: 10% RL penetration,  5 max controllable vehicles
# - 1: 25% RL penetration, 13 max controllable vehicles
# - 2: 33% RL penetration, 17 max controllable vehicles
EXP_NUM = 0

# time horizon of a single rollout
HORIZON = 600
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 1

# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = [5, 13, 17][EXP_NUM]

# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 1
additional_net_params["pre_merge_length"] = 500

# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2,
        "v0": 20,           # 목표 속도
        "T": 1,             # 시간 간격
        "a": 1.5,           # 최대 가속도
        "b": 1.5,           # 편안한 감속도
    }),
    car_following_params=SumoCarFollowingParams(
        speed_mode=7,
        accel=1.5,          # 가속도 제한
        decel=1.5,          # 감속도 제한
    ),
    num_vehicles=5)      # 초기 IDM 차량
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,
    ),
    num_vehicles=0)      # 초기 RL 차량은 0

# 신호등 설정
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

# 교통 흐름 설정
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=100,
    departLane="free",
    departSpeed=7.5)

# flow_params 정의
flow_params = dict(
    exp_tag="merge_with_lights",
    env_name=MyCustomMergePOEnv,   # 기본 MultiAgentMergePOEnv 대신 우리가 만든 커스텀 환경
    network=MergeNetwork,
    simulator="traci",
    sim=SumoParams(
        sim_step=0.2,
        render=False,
        restart_instance=True,
    ),
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=5,
        warmup_steps=0,
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "normalize_obs": True,
            "clip_actions": True,
            "obs_range": [-100, 100],
        },
    ),
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
    ),
    veh=vehicles,
    initial=InitialConfig(),
    tls=traffic_lights
)

# 환경 생성 및 등록
create_env, env_name = make_create_env(params=flow_params, version=0)
register_env(env_name, create_env)
test_env = create_env()

# Observation/Action spaces
obs_space = test_env.observation_space
act_space = test_env.action_space

def gen_policy():
    """Generate a policy in RLlib with observation normalization."""
    return PPOTFPolicy, obs_space, act_space, {
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "tanh",
            "vf_share_layers": True,
            "preprocessor_pref": None,
        },
        "gamma": 0.99,
        "lr": 5e-5,
        "num_sgd_iter": 10,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.01,
        "observation_filter": "MeanStdFilter",
    }

# 추가로 warmup_steps수정
flow_params["env"] = EnvParams(
    horizon=HORIZON,
    sims_per_step=5,
    warmup_steps=50,
    additional_params={
        "max_accel": 1.5,
        "max_decel": 1.5,
        "target_velocity": 20,
        "normalize_obs": True,
        "clip_actions": True,
    },
)

# 최종 환경 재생성 후 등록
create_env, env_name = make_create_env(params=flow_params, version=0)
register_env(env_name, create_env)
test_env = create_env()

obs_space = test_env.observation_space
act_space = test_env.action_space