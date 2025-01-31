"""Example of a figure 8 network with human-driven vehicles and RL training."""

import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env

from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, RLController
from flow.core.params import SumoParams, EnvParams, NetParams, VehicleParams, SumoCarFollowingParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import MultiAgentAccelPOEnv
from flow.networks import FigureEightNetwork
from flow.core.params import TrafficLightParams  # 신호등 설정 추가

# 🚀 RL 학습 파라미터 설정
HORIZON = 1500  # 한 번의 rollout이 지속되는 시간 (step)
N_ROLLOUTS = 20  # 한 번의 학습 iteration 동안 실행되는 rollout 개수
N_CPUS = 2  # 병렬 실행할 작업자 수

TARGET_VELOCITY = 20  # 목표 속도 (m/s)
MAX_ACCEL = 3  # RL 차량 최대 가속도
MAX_DECEL = 3  # RL 차량 최대 감속도

NUM_AUTOMATED = 1  # RL 차량 개수 (1, 2, 7, 14 중 선택 가능)
assert NUM_AUTOMATED in [1, 2, 7, 14], "NUM_AUTOMATED 값이 유효하지 않습니다."

# 🚀 Ray 초기화 (중복 실행 방지)
ray.init(ignore_reinit_error=True)

vehicles = VehicleParams()

# RL 차량 추가 (NUM_AUTOMATED 만큼 추가)
for i in range(NUM_AUTOMATED):
    vehicles.add(
        veh_id=f"rl_{i}",
        acceleration_controller=(RLController, {}),
        lane_change_controller=(StaticLaneChanger, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=31,
            accel=MAX_ACCEL,
            decel=MAX_DECEL,
        ),
        num_vehicles=1
    )

# 기존 IDM 차량 추가
vehicles.add(
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

# 신호등 설정 추가
traffic_lights = TrafficLightParams(baseline=False)
traffic_lights.add(
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

flow_params = dict(
    exp_tag='figure8_with_rl',
    env_name=MultiAgentAccelPOEnv,
    network=FigureEightNetwork,
    simulator='traci',
    sim=SumoParams(render=True, sim_step=0.1),
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": TARGET_VELOCITY,
            "max_accel": MAX_ACCEL,
            "max_decel": MAX_DECEL,
            "sort_vehicles": False
        },
    ),
    net=NetParams(additional_params=ADDITIONAL_NET_PARAMS.copy()),
    veh=vehicles,
    tls=traffic_lights
)

# 환경이 이미 등록되어 있는지 확인 후 등록
env_id = "MultiAgentAccelPOEnv-v0"
if env_id not in gym.envs.registry.env_specs:
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

# 🚀 PPOTrainer로 RL 학습 실행
trainer = PPOTrainer(env=env_name, config={
    "num_workers": N_CPUS,
    "train_batch_size": HORIZON * N_ROLLOUTS,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 10,
})

# RL 학습 10회 반복
for i in range(10):
    result = trainer.train()
    mean_reward = result.get("episode_reward_mean", 0)  # 에러 방지
    print(f"Iteration {i}, reward: {mean_reward}")

# 학습된 정책 저장
checkpoint_path = trainer.save()
print(f"Checkpoint saved at {checkpoint_path}")

print(f"Traffic light parameters: {flow_params['tls']}")
print("Flow parameters:", flow_params)
