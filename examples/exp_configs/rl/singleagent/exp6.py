"""Ring road example with traffic lights."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, TrafficLightParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, RLController
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.envs import WaveAttenuationPOEnv
from flow.networks import RingNetwork

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 1

# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()

# Add idm vehicles
vehicles.add(
    veh_id='idm',
    acceleration_controller=(IDMController, {"noise": 0.2}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=7,
        decel=2.5,
        min_gap=2.5,    # 최소 차간 거리 증가
        tau=1.0         # 시간 간격 추가
    ),
    initial_speed=0,
    num_vehicles=21
)

# Add the RL-controlled vehicle (using the RLController and StaticLaneChanger)
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=31,  # using the original value (you can change this if needed)
        decel=2.5,
    ),
)

# Define traffic light settings.
traffic_lights = TrafficLightParams(baseline=False)
traffic_lights.add(
    node_id="bottom",  # Figure Eight의 교차점 노드
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
    exp_tag="stabilizing_the_ring",

    # name of the flow environment the experiment is running on
    env_name=WaveAttenuationPOEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=False
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [220, 270],
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 260,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),

    # Include the traffic light settings.
    tls=traffic_lights
)
