from flow.envs.base import Env
from gym import spaces
import numpy as np

class TrafficLightFigureEightEnv(Env):
    """Custom environment for Figure Eight Network with static traffic lights."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the environment."""
        super().__init__(env_params, sim_params, network, simulator)

        # 시뮬레이터 객체 초기화
        self.sim = self.k

        # sim_step 값을 가져오기
        self.sim_step = sim_params.sim_step  # sim_step 값을 sim_params에서 가져옴

        # 트래픽 라이트 초기 상태
        self.current_phase = 0
        self.phase_time = 0
        self.phases = [
            {"duration": 15, "state": "GrGr"},
            {"duration": 3, "state": "yrGr"},
            {"duration": 2, "state": "rrrr"},
            {"duration": 15, "state": "rGrG"},
            {"duration": 3, "state": "ryrG"},
            {"duration": 2, "state": "rrrr"}
        ]

    @property
    def action_space(self):
        """Gym에서 요구하는 action_space 프로퍼티."""
        return spaces.Discrete(1)  # Only one possible action (no-op)

    @property
    def observation_space(self):
        """Gym에서 요구하는 observation_space 프로퍼티."""
        return spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.env_params.additional_params["num_observed"],),
            dtype=np.float32
        )

    def reset(self):
        """Reset the environment to initial state."""
        print("Resetting the environment...")  # 디버깅 로그 추가
        super().reset()  # 시뮬레이션 리셋(차량 재배치 등)

        # 신호등 초기화
        self.reset_traffic_lights()

        # 초기 관측치 반환
        state = self.get_state()
        print(f"Initial state: {state}")  # 초기 상태 출력
        return state


    def _apply_rl_actions(self, actions):
        """No-op since traffic lights are static."""
        print(f"Applying RL actions: {actions}")  # 디버깅 로그 추가
        self._update_traffic_lights()


    def _update_traffic_lights(self):
        """Update traffic lights based on the current phase and time."""
        print(f"Updating traffic lights... Current phase: {self.current_phase}")  # 디버깅 로그 추가
        self.phase_time += self.sim_step
        if self.phase_time >= self.phases[self.current_phase]["duration"]:
            self.phase_time = 0
            self.current_phase = (self.current_phase + 1) % len(self.phases)
            self.k.traffic_light.set_state("center0", self.phases[self.current_phase]["state"])
            print(f"Traffic light 'center0' set to state: {self.phases[self.current_phase]['state']}")  # 신호 변경 로그 추가

        
    def reset_traffic_lights(self):
        """Reset traffic lights to initial phase."""
        self.k.traffic_light.set_state("center0", self.phases[0]["state"])
        self.current_phase = 0
        self.phase_time = 0

    def get_state(self):
        """Get the current state of the environment (ex: 차량 속도들)."""
        observed_ids = self.k.vehicle.get_ids()
        speeds = [
            self.k.vehicle.get_speed(veh_id)
            for veh_id in observed_ids[:self.env_params.additional_params["num_observed"]]
        ]
        return np.array(speeds, dtype=np.float32)

    def compute_reward(self, obs, fail=None):
        """Compute the reward for the current state."""
        if fail:
            return -100  # 실패 시 큰 패널티 부여 (값은 필요에 따라 조정)
        if obs is None:
            return -100  
    
        return -np.sum(obs)