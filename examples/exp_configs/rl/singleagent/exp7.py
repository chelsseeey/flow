"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
on-ramp merge to a single lane open highway network.
"""

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig
from flow.core.params import NetParams, InFlows, SumoCarFollowingParams, TrafficLightParams
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController
from flow.envs.multiagent import MultiAgentMergePOEnv
from flow.networks import MergeNetwork
from flow.utils.registry import make_create_env

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
        "v0": 20,           # 목표 속도 지정
        "T": 1,             # 시간 간격
        "a": 1.5,           # 최대 가속도
        "b": 1.5,           # 편안한 감속도
    }),
    car_following_params=SumoCarFollowingParams(
        speed_mode=7,
        accel=1.5,          # 가속도 제한
        decel=1.5,          # 감속도 제한
    ),
    num_vehicles=5)      # 초기에 5대의 IDM 차량
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,
    ),
    num_vehicles=0)      # 초기에 0대의 RL 차량

# 신호등 설정 추가
traffic_lights = TrafficLightParams(baseline=False)
traffic_lights.add(
    node_id="center",  # 신호등 노드
    tls_type="static",  # 고정 주기 신호등
    programID="1",  # 프로그램 ID
    phases=[
        {"duration": "5", "state": "GrGr"},  # 가로 초록
        {"duration": "3", "state": "yrGr"},   # 가로 노랑
        {"duration": "2", "state": "rrrr"},   # 모두 빨강
        {"duration": "5", "state": "rGrG"},  # 세로 초록
        {"duration": "3", "state": "ryrG"},   # 세로 노랑
        {"duration": "2", "state": "rrrr"}    # 모두 빨강
    ]
)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
# 고속도로 진입 차량
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE, # 1800 vehicles/hour (90%)
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE, # 200 vehicles/hour (10%)
    departLane="free",
    departSpeed=10)
# 합류 지점 진입 차량
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=100,   # 100 vehicles/hour
    departLane="free",
    departSpeed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag='merge_with_lights',

    # name of the flow environment the experiment is running on
    env_name=MultiAgentMergePOEnv,

    # name of the network class the experiment is running on
    network=MergeNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.2,
        render=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=5,
        warmup_steps=0,
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "normalize_obs": True,    # 관찰값 정규화 활성화
            "clip_actions": True,     # 행동값 클리핑 추가
            "obs_range": [-100, 100],     # 관찰값 범위 제한

        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),

    tls=traffic_lights
)

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

# test_env 생성 전에 custom observation space 정의
custom_obs_space = Box(
    low=np.array([-2, -2, -2, 0, -2]),    # 최소값
    high=np.array([2, 2, 2, 40, 2]),      # 최대값
    dtype=np.float32
)

# test_env 생성 후 observation space 교체
test_env = create_env()
test_env.observation_space = custom_obs_space  # 커스텀 observation space 적용
act_space = test_env.action_space

def gen_policy():
    """Generate a policy in RLlib with observation normalization."""
    return PPOTFPolicy, obs_space, act_space, {
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "tanh",
            # custom_preprocessor 제거
            "vf_share_layers": True,
            "preprocessor_pref": None,  # 기본 전처리기 사용
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
        "observation_filter": "MeanStdFilter",  # NoFilter 대신 MeanStdFilter 사용
        "normalize_observations": True,  # 관찰값 정규화 활성화
        "normalize_rewards": False,      # 보상 정규화 비활성화
    }

# 환경 파라미터도 수정
flow_params["env"] = EnvParams(
    horizon=HORIZON,
    sims_per_step=5,
    warmup_steps=50,  # 워밍업 스텝 추가
    additional_params={
        "max_accel": 1.5,
        "max_decel": 1.5,
        "target_velocity": 20,
        "normalize_obs": True,
        "clip_actions": True,
        # observation_normalizer 제거하고 단순화
        "obs_space": custom_obs_space,  # 커스텀 observation space 사용
    },
)