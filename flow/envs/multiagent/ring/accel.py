"""Environment for training the acceleration behavior of vehicles in a ring."""
import numpy as np
from gym.spaces import Box

from flow.core import rewards
from flow.envs.ring.accel import AccelEnv
from flow.envs.multiagent.base import MultiEnv


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 20,
    # collision penalty for reward calculation
    "collision_penalty": 10
}


class AdversarialAccelEnv(AccelEnv, MultiEnv):
    """Adversarial multi-agent acceleration env.

    States
        The observation of both the AV and adversary agent consist of the
        velocities and absolute position of all vehicles in the network. This
        assumes a constant number of vehicles.

    Actions
        * AV: The action space of the AV agent consists of a vector of bounded
          accelerations for each autonomous vehicle. In order to ensure safety,
          these actions are further bounded by failsafes provided by the
          simulator at every time step.
        * Adversary: The action space of the adversary agent consists of a
          vector of perturbations to the accelerations issued by the AV agent.
          These are directly added to the original accelerations by the AV
          agent.

    Rewards
        * AV: The reward for the AV agent is equal to the mean speed of all
          vehicles in the network.
        * Adversary: The adversary receives a reward equal to the negative
          reward issued to the AV agent.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]
        av_action = rl_actions['av']
        adv_action = rl_actions['adversary']
        perturb_weight = self.env_params.additional_params['perturb_weight']
        rl_action = av_action + perturb_weight * adv_action
        self.k.vehicle.apply_acceleration(sorted_rl_ids, rl_action)

    def compute_reward(self, rl_actions, **kwargs):
        """Compute opposing rewards for agents.

        The agent receives the class definition reward,
        the adversary receives the negative of the agent reward
        """
        if self.env_params.evaluate:
            reward = np.mean(self.k.vehicle.get_speed(
                self.k.vehicle.get_ids()))
            return {'av': reward, 'adversary': -reward}
        else:
            reward = rewards.desired_velocity(self, fail=kwargs['fail'])
            return {'av': reward, 'adversary': -reward}

    def get_state(self, **kwargs):
        """See class definition for the state.

        The adversary state and the agent state are identical.
        """
        state = np.array([[
            self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed(),
            self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
        ] for veh_id in self.sorted_ids])
        state = np.ndarray.flatten(state)
        return {'av': state, 'adversary': state}


class MultiAgentAccelPOEnv(MultiEnv):
    """Multi-agent partially observable acceleration environment with collision detection."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the environment.
        
        Parameters
        ----------
        env_params : flow.core.params.EnvParams
            환경 파라미터
        sim_params : flow.core.params.SimParams
            시뮬레이션 파라미터
        network : flow.networks.base.Network
            교통 네트워크
        simulator : str, optional
            사용할 시뮬레이터, defaults to 'traci'
        """
        required_params = [
            "max_accel",
            "max_decel", 
            "target_velocity",
            "collision_penalty"
        ]
        
        for p in required_params:
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.leader = []
        self.follower = []
        self.collision_counts = 0
        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """Return the observation space."""
        return Box(low=-5, high=5, shape=(7,), dtype=np.float32)

    @property
    def action_space(self):
        """Return the action space."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),
            dtype=np.float32)

    def get_state(self, **kwargs):
        """Return the state of the simulation.
        
        Returns
        -------
        numpy.ndarray or dict
            차량의 상태 정보를 포함하는 observation
        """
        self.leader = []
        self.follower = []
        obs = {}

        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_pos = self.k.vehicle.get_x_by_id(rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)

            # 선행 차량 정보
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id in ["", None]:
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                           - self.k.vehicle.get_x_by_id(rl_id) \
                           - self.k.vehicle.get_length(rl_id)

            # 후행 차량 정보
            follower = self.k.vehicle.get_follower(rl_id)
            if follower in ["", None]:
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            obs[rl_id] = np.array([
                this_pos / max_length,
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length,
                self.collision_counts
            ])

        return list(obs.values())[0] if len(obs) == 1 else obs

    def step(self, rl_actions):
        """Execute one step of the environment.
        
        Parameters
        ----------
        rl_actions : dict
            각 RL 차량의 가속도 행동
            
        Returns
        -------
        state : numpy.ndarray or dict
            새로운 observation
        rewards : dict
            각 에이전트의 보상
        done : bool
            에피소드 종료 여부
        info : dict
            추가 정보
        """
        try:
            colliding_vehicles = self.k.kernel_api.simulation.getCollidingVehiclesIDList()
            collision_count = self.k.kernel_api.simulation.getCollidingVehiclesNumber()
            
            if collision_count > 0:
                self.collision_counts += collision_count
        except:
            # Fallback to headway-based collision detection
            colliding_vehicles = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_headway(veh_id) <= 0:
                    colliding_vehicles.append(veh_id)
            collision_count = len(colliding_vehicles)
            if collision_count > 0:
                self.collision_counts += collision_count

        for _ in range(self.env_params.sims_per_step):
            self._apply_rl_actions(rl_actions)
            self.k.simulation.simulation_step()

        states = self.get_state()
        rewards = self.compute_reward(rl_actions, collisions=collision_count)
        done = self.check_termination()
        
        info = {
            'collision_count': self.collision_counts,
            'new_collisions': collision_count,
            'colliding_vehicles': colliding_vehicles
        }

        return states, rewards, done, info

    def _apply_rl_actions(self, rl_actions):
        """Apply acceleration actions from RL agents."""
        if rl_actions:
            for rl_id, acceleration in rl_actions.items():
                self.k.vehicle.apply_acceleration(rl_id, acceleration)

    def compute_reward(self, rl_actions, **kwargs):
        """Calculate reward with collision penalty.
        
        Parameters
        ----------
        rl_actions : dict
            각 RL 차량의 행동
        **kwargs : dict
            추가 키워드 인자
            
        Returns
        -------
        dict
            각 에이전트의 보상
        """
        if rl_actions is None:
            return {}

        rewards_dict = {}
        for rl_id in rl_actions.keys():
            # Flow의 rewards 모듈 사용
            reward = rewards.desired_velocity(self, fail=kwargs.get('fail', False))
            
            # 충돌 페널티 적용
            collision_penalty = self.env_params.additional_params.get('collision_penalty', 10)
            if 'collisions' in kwargs:
                penalty = kwargs['collisions'] * collision_penalty
                reward = reward - penalty
                
            rewards_dict[rl_id] = reward

        return rewards_dict

    def reset(self):
        """Reset the environment state."""
        self.collision_counts = 0
        self.leader = []
        self.follower = []
        return super().reset()

    def additional_command(self):
        """Execute additional commands for each time step."""
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)
            follow_id = self.k.vehicle.get_follower(rl_id) or rl_id
            self.k.vehicle.set_observed(follow_id)