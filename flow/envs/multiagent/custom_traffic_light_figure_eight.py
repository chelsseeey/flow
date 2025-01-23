# flow/envs/custom/custom_traffic_light_figure_eight.py

from flow.envs.base import Env
from gym import spaces
import numpy as np

class TrafficLightFigureEightEnv(Env):
    """Custom environment for Figure Eight Network with static traffic lights."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the environment."""
        super().__init__(env_params, sim_params, network, simulator)

        # (1) 내부 변수에 action_space / observation_space 저장
        self._action_space = spaces.Discrete(1)  # Only one possible action (no-op)
        self._observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.env_params.additional_params["num_observed"],),
            dtype=np.float32
        )

        # (2) 시뮬레이터 객체 초기화
        self.sim = self.k

        # (3) 트래픽 라이트 초기 상태
        self.current_phase = 0

    # (4) Gym에서 요구하는 추상 프로퍼티 구현 (@property)
    @property
    def action_space(self):
        """Gym에서 요구하는 action_space 프로퍼티."""
        return self._action_space

    @property
    def observation_space(self):
        """Gym에서 요구하는 observation_space 프로퍼티."""
        return self._observation_space

    def reset(self):
        """Reset the environment to initial state."""
        super().reset()  # 시뮬레이션 리셋(차량 재배치 등)

        # 신호등 초기화
        self.reset_traffic_lights()

        # 초기 관측치 반환
        return self.get_state()

    def step(self, action):
        """Run one timestep of the environment's dynamics."""
        # static하므로 action을 사용하지 않음
        self.sim.step()

        obs = self.get_state()
        reward = self.compute_reward(obs)
        done = self.sim.get_time() >= self.env_params.horizon
        info = {}

        return obs, reward, done, info

    def get_state(self):
        """Get the current state of the environment (ex: 차량 속도들)."""
        observed_ids = self.k.vehicle.get_ids()
        speeds = [
            self.k.vehicle.get_speed(veh_id)
            for veh_id in observed_ids[:self.env_params.additional_params["num_observed"]]
        ]
        return np.array(speeds, dtype=np.float32)

    def compute_reward(self, obs):
        """Compute the reward for the current state."""
        # 예: 전체 속도의 합을 줄이는 방향 -> -np.sum(...)
        return -np.sum(obs)

    def reset_traffic_lights(self):
        """Reset traffic lights to initial phase."""
        node_id = "center0"  # 트래픽 라이트 등록 시 사용한 노드 ID와 동일
        all_tl_props = self.network.traffic_lights.get_properties()
        # 형태: { "center0": {"phases": [...], ...}, "other_node": {...}, ... }

        if node_id in all_tl_props and "phases" in all_tl_props[node_id]:
            phases = all_tl_props[node_id]["phases"]
            if phases:
                initial_phase = phases[0]["state"]
            else:
                # phases가 비어있다면 기본값
                initial_phase = "GrGr"
        else:
            # node_id가 없거나 phases 키가 없다면
            initial_phase = "GrGr"

        self.k.traffic_light.set_state(node_id, initial_phase)
        self.current_phase = 0

    def _apply_rl_actions(self, actions):
        """No-op since traffic lights are static."""
        pass
