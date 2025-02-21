from flow.core.experiment import Experiment
from flow.envs.multiagent.ring.accel import MultiAgentAccelPOEnv
from flow.networks import RingNetwork
from flow.core.params import (
    VehicleParams,
    NetParams,
    InitialConfig,
    EnvParams,
    SumoParams
)

def create_env():
    # 네트워크 파라미터
    net_params = NetParams(
        additional_params={
            "length": 230,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        },
    )

    # 차량 파라미터
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(None, {}),
        car_following_params=None,
        num_vehicles=5
    )
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(None, {}),
        car_following_params=None,
        num_vehicles=2
    )

    # 환경 파라미터
    env_params = EnvParams(
        additional_params={
            "max_accel": 3,
            "max_decel": 3,
            "target_velocity": 10,
            "collision_penalty": 10,
        },
        horizon=1000,
    )

    # 초기 설정
    initial_config = InitialConfig(spacing="uniform")

    # SUMO 파라미터
    sim_params = SumoParams(sim_step=0.1, render=True)  # render=True로 설정하여 시각화

    # 네트워크 생성
    network = RingNetwork(
        name="ring",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    # 환경 생성
    env = MultiAgentAccelPOEnv(
        env_params=env_params,
        sim_params=sim_params,
        network=network
    )
    
    return env

def monitor_collisions():
    env = create_env()
    env.reset()
    
    print("Starting collision monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # 랜덤 액션으로 환경 실행
            action = {rl_id: [1.0] for rl_id in env.k.vehicle.get_rl_ids()}
            _, _, _, info = env.step(action)
            
            # 충돌 정보 출력
            for rl_id, agent_info in info.items():
                if agent_info['new_collisions'] > 0:
                    print(f"\nTime: {env.k.simulation.step_counter}")
                    print(f"Vehicle {rl_id}: New collisions = {agent_info['new_collisions']}")
                    print(f"Colliding vehicles: {agent_info['colliding_vehicles']}")
                    print(f"Total collisions: {agent_info['collision_count']}")
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        env.terminate()

if __name__ == "__main__":
    monitor_collisions()