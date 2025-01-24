"""Example of a figure 8 network with human-driven vehicles.

Right-of-way dynamics near the intersection causes vehicles to queue up on
either side of the intersection, leading to a significant reduction in the
average speed of vehicles in the network.
"""
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs import AccelEnv
from flow.networks import FigureEightNetwork
from flow.core.params import TrafficLightParams  # 신호등 설정 추가
from flow.controllers.rlcontroller import RLController


vehicles = VehicleParams()

# RL 차량 추가 (1대)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),  # RLController 추가
    lane_change_controller=(StaticLaneChanger, {}),  # 필요 시 RL 컨트롤러로 변경
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,
        decel=1.5,
    ),
    num_vehicles=1  # RL 차량 1대
)

# 기존 IDM 차량 추가
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,
        decel=1.5,
    ),
    initial_speed=0,
    num_vehicles=13
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
    exp_tag='figure8_with_lights',  # 실험 이름 변경

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=FigureEightNetwork,

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
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # 신호등 설정 반영
    tls=traffic_lights
)

print(f"Traffic light parameters: {flow_params['tls']}")

# flow_params 딕셔너리 출력
print("Flow parameters:", flow_params)
