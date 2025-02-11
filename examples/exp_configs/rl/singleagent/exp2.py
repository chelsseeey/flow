"""Figure eight example with traffic lights."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, TrafficLightParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import MultiAgentAccelPOEnv  # 변경
from flow.networks import FigureEightNetwork
import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
plt.ion()

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 1

# Define the vehicles in the network.
# Here we use one RL-controlled vehicle and 13 human-driven vehicles.
vehicles = VehicleParams()

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

# 콜백 클래스 정의
class CollisionCallback:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_xlabel('Iteration Number')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Collisions per Iteration')
        self.ax.grid(True)
        self.iteration_collisions = []
        
    def __call__(self, env, worker):
        collision_count = 0
        for _ in range(N_ROLLOUTS):
            f = io.StringIO()
            with redirect_stdout(f):
                state = env.reset()
                done = False
                while not done:
                    action = env.action_space.sample()
                    next_state, reward, done, info = env.step(action)
                    
            output = f.getvalue()
            if "Collision detected at time step" in output:
                collision_count += 1
                
        self.iteration_collisions.append(collision_count)
        
        # 그래프 업데이트
        self.ax.clear()
        self.ax.set_xlabel('Iteration Number')
        self.ax.set_ylabel('Number of Collisions')
        self.ax.set_title('Collisions per Iteration')
        self.ax.grid(True)
        self.ax.plot(range(1, len(self.iteration_collisions) + 1), 
                    self.iteration_collisions, 'g-')
        plt.pause(0.01)

# 콜백 추가
flow_params['callback'] = CollisionCallback()

print("Traffic light parameters:", flow_params['tls'])
print("Flow parameters:", flow_params)
