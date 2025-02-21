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
    """Multi-agent partially observable acceleration environment."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the environment."""
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # used to store the leader and follower IDs of RL vehicles
        self.leader = []
        self.follower = []
        
        # collision monitoring 변수 추가
        self.collision_counts = 0

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """Return the observation space (collision count 포함)."""
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
        """Return the state of the simulation."""
        self.leader = []
        self.follower = []
        obs = {}

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_pos = self.k.vehicle.get_x_by_id(rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)

            # get leader
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

            # get follower
            follower = self.k.vehicle.get_follower(rl_id)
            if follower in ["", None]:
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            # collision count를 포함한 observation
            obs[rl_id] = np.array([
                this_pos / max_length,
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length,
                self.collision_counts  # collision count 추가
            ])

        if len(obs) == 1:
            return list(obs.values())[0]
        else:
            return obs

    def _apply_rl_actions(self, rl_actions):
        """Apply the acceleration actions from the RL agents."""
        if rl_actions:
            for rl_id, acceleration in rl_actions.items():
                self.k.vehicle.apply_acceleration(rl_id, acceleration)

    def step(self, rl_actions):
        """Execute one step of the environment."""
        # collision check using TraCI
        collisions = self.k.kernel_api.simulation.getCollisions()
        if collisions:
            self.collision_counts += len(collisions)

        for _ in range(self.env_params.sims_per_step):
            self._apply_rl_actions(rl_actions)
            self.k.simulation.simulation_step()

        # update state, calculate rewards
        states = self.get_state()
        rewards = self.compute_reward(rl_actions, collisions=len(collisions))
        done = self.check_termination()
        
        # collision info 추가
        info = {
            'collision_count': self.collision_counts,
            'new_collisions': len(collisions) if collisions else 0
        }

        return states, rewards, done, info

    def compute_reward(self, rl_actions, **kwargs):
        """Calculate reward with collision penalty."""
        if rl_actions is None:
            return {}

        # 기본 reward 계산
        rewards = {}
        for rl_id in rl_actions.keys():
            reward = rewards.desired_velocity(self, fail=kwargs.get('fail', False))
            
            # collision penalty 적용
            collision_penalty = self.env_params.additional_params.get('collision_penalty', 10)
            if 'collisions' in kwargs:
                penalty = kwargs['collisions'] * collision_penalty
                reward = reward - penalty
                
            rewards[rl_id] = reward

        return rewards

    def reset(self):
        """Reset the environment."""
        self.collision_counts = 0
        self.leader = []
        self.follower = []
        return super().reset()

    def additional_command(self):
        """See parent class."""
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)
            # follower
            follow_id = self.k.vehicle.get_follower(rl_id) or rl_id
            self.k.vehicle.set_observed(follow_id)