"""
Example of a figure 8 network with human-driven vehicles.
Right-of-way dynamics near the intersection causes vehicles to queue up on
either side of the intersection, leading to a significant reduction in the
average speed of vehicles in the network.
"""
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, GridRouter
from flow.core.params import SumoParams, EnvParams, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent.custom_traffic_light_figure_eight import TrafficLightFigureEightEnv  # 커스텀 Env 임포트
from flow.networks import FigureEightNetwork
from flow.core.params import TrafficLightParams


# 1. grid_array 수정 - grid 환경에 맞게 설정
ADDITIONAL_NET_PARAMS.update({
    "grid_array": {
        "short_length": 300,
        "inner_length": 300,
        "long_length": 500,
        "row_num": 1,        # 1x1 grid로 변경 (8자형 교차로와 비슷하게)
        "col_num": 1,        # 1x1 grid로 변경
        "cars_left": 5,      # 각 방향 차량 수 조정
        "cars_right": 5,
        "cars_top": 5,
        "cars_bot": 5
    },
    # Figure8Network 파라미터도 유지
    "radius_ring": 30,
    "lanes": 1,
    "speed_limit": 30,
    "resolution": 40
})

# 차량 설정
vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=0,
        decel=3.0,
    ),
    initial_speed=0,
    num_vehicles=14
)

# 신호등 설정
traffic_lights = TrafficLightParams()
traffic_lights.add(
    node_id="center0",  # Figure Eight의 교차점 노드 ID
    programID=1,        # traffic_light_grid.py와 동일한 정수 ID 사용
    tls_type="static",  # 고정 주기 신호등
    phases=[
        {"duration": "15", "state": "GrGr"},  # 가로 초록
        {"duration": "3", "state": "yrGr"},   # 가로 노랑
        {"duration": "2", "state": "rrrr"},   # 모두 빨강
        {"duration": "15", "state": "rGrG"},  # 세로 초록
        {"duration": "3", "state": "ryrG"},   # 세로 노랑
        {"duration": "2", "state": "rrrr"}    # 모두 빨강
    ]
)

# 시뮬레이션 파라미터 설정
flow_params = dict(
    exp_tag='figure8_with_lights',
    env_name=TrafficLightFigureEightEnv,
    network=FigureEightNetwork,
    simulator='traci',
    sim=SumoParams(
        render=True,
        sim_step=0.1,

    ),
    env=EnvParams(
        horizon=1500,
        additional_params={
            "switch_time": 3,
            "tl_type": "static",
            "discrete": False,
            "num_observed": 10,
            "target_velocity": 30,
            "max_accel": 3.0,
            "max_decel": 3.0,
            "action_space": "continuous"
        },
    ),
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS
    ),
    veh=vehicles,
    traffic_lights=traffic_lights  # traffic_lights를 별도의 매개변수로 전달
)

# 네트워크 정보 출력
print("차선 수:", ADDITIONAL_NET_PARAMS["lanes"])
print("원형 구간 반지름:", ADDITIONAL_NET_PARAMS["radius_ring"])
print("속도 제한:", ADDITIONAL_NET_PARAMS["speed_limit"])
print(f"Environment being used: {flow_params['env_name']}")