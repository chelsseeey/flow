"""Example of a merge network with human-driven vehicles.

In the absence of autonomous vehicles, the network exhibits properties of
convective instability, with perturbations propagating upstream from the merge
point before exiting the network.
"""

from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows, SumoCarFollowingParams, TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController
from flow.envs.merge import MergePOEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import MergeNetwork

# experiment number
# - 0: 10% RL penetration,  5 max controllable vehicles
# - 1: 25% RL penetration, 13 max controllable vehicles
# - 2: 33% RL penetration, 17 max controllable vehicles
EXP_NUM = 0


# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",     # vehicle ID "human"으로 고정!
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,
    ),
    num_vehicles=5)
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


inflow = InFlows()
# 고속도로 진입 차량
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=FLOW_RATE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE, # 200 vehicles/hour (10%)
    departLane="free",
    departSpeed=10,
    speed_mode=31)
# 합류 지점 진입 차량
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=100,
    departLane="free",
    departSpeed=7.5)


flow_params = dict(
    # name of the experiment
    exp_tag='merge_with_lights',

    # name of the flow environment the experiment is running on
    env_name=MergePOEnv,

    # name of the network class the experiment is running on
    network=MergeNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        emission_path="./data/",
        sim_step=0.2,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=3600,
        additional_params=ADDITIONAL_ENV_PARAMS,
        sims_per_step=5,
        warmup_steps=0,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params={
            "merge_length": 100,
            "pre_merge_length": 500,
            "post_merge_length": 100,
            "merge_lanes": 1,
            "highway_lanes": 1,
            "speed_limit": 30,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        perturbation=5.0,
    ),

    tls=traffic_lights
)
