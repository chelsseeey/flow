"""
Used as an example of ring experiment.

"""
from flow.controllers import IDMController, SimLaneChangeController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs import AccelEnv
from flow.networks import RingNetwork
from flow.core.params import TrafficLightParams  # 신호등 설정 추가
from flow.controllers.rlcontroller import RLController



vehicles = VehicleParams()

# RL 차량 추가 (1대)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),  # RLController 추가
    lane_change_controller=(SimLaneChangeController, {}),  # 필요 시 RL 컨트롤러로 변경
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,
        decel=2.5,
    ),
    num_vehicles=1  # RL 차량 1대
)

# 기존 IDM 차량 추가 (21대)
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=7,
        decel=2.5,
    ),
    initial_speed=0,
    num_vehicles=21
)

# 신호등 설정 추가
traffic_lights = TrafficLightParams(baseline=False)
traffic_lights.add(
    node_id="center",  # Figure Eight의 교차점 노드
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

flow_params = dict(
    # name of the experiment
    exp_tag='ring_with_lights',

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.1,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=100000000,
        additional_params=ADDITIONAL_ENV_PARAMS,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        bunching=20

    ),
    
    tls=traffic_lights
)
