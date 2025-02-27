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
    """Multi-agent partially observable acceleration environment with OBB collision detection."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the environment."""
        import logging
        
        # Logger 설정
        self.logger = logging.getLogger('CollisionMonitor')
        self.logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - Collisions: %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        fh = logging.FileHandler('collision_log.txt')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        required_params = [
            "max_accel", "max_decel", 
            "target_velocity", "collision_penalty"
        ]
        
        for p in required_params:
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.leader = []
        self.follower = []
        self.collision_counts = 0
        super().__init__(env_params, sim_params, network, simulator)

    def detect_obb_collision(self, veh1, veh2):
        """OBB 충돌 감지"""
        try:
            # 차량 정보 획득 (x, y 좌표를 getPosition으로 얻기)
            pos1 = self.k.vehicle.get_position(veh1)  # (x, y) tuple 반환
            pos2 = self.k.vehicle.get_position(veh2)
            x1, y1 = pos1
            x2, y2 = pos2
            
            angle1 = np.radians(self.k.vehicle.get_angle(veh1))
            angle2 = np.radians(self.k.vehicle.get_angle(veh2))
            
            def get_corners(x, y, length, width, angle):
                corners = np.array([
                    [-length/2, -width/2],
                    [length/2, -width/2],
                    [length/2, width/2],
                    [-length/2, width/2]
                ])
                
                rotation = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])
                
                corners = np.dot(corners, rotation.T)
                corners += np.array([x, y])
                return corners
            
            corners1 = get_corners(x1, y1, 
                                self.k.vehicle.get_length(veh1),
                                self.k.vehicle.get_width(veh1), 
                                angle1)
            corners2 = get_corners(x2, y2,
                                self.k.vehicle.get_length(veh2),
                                self.k.vehicle.get_width(veh2),
                                angle2)
            
            def get_axes(corners):
                axes = []
                for i in range(4):
                    p1 = corners[i]
                    p2 = corners[(i + 1) % 4]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    axes.append(normal / np.linalg.norm(normal))
                return axes
            
            axes = get_axes(corners1) + get_axes(corners2)
            for axis in axes:
                proj1 = [np.dot(corner, axis) for corner in corners1]
                proj2 = [np.dot(corner, axis) for corner in corners2]
                
                min1, max1 = min(proj1), max(proj1)
                min2, max2 = min(proj2), max(proj2)
                
                if max1 < min2 or max2 < min1:
                    return False
                    
            return True
        except Exception as e:
            self.logger.error(f"Error in detect_obb_collision: {e}")
            return False

    def detect_collisions(self):
        """모든 차량 쌍에 대해 OBB 충돌 감지 수행"""
        colliding_vehicles = []
        vehicles = self.k.vehicle.get_ids()
        
        for i, veh1 in enumerate(vehicles):
            for veh2 in vehicles[i+1:]:
                if self.detect_obb_collision(veh1, veh2):
                    colliding_vehicles.extend([veh1, veh2])
                    self.logger.info(f"OBB Collision detected between {veh1} and {veh2}")
        
        return list(set(colliding_vehicles))

    def step(self, rl_actions):
        """Execute one step of the environment."""
        # OBB 충돌 감지
        colliding_vehicles = self.detect_collisions()
        collision_count = len(colliding_vehicles) // 2  # 각 충돌은 2개의 차량을 포함
        
        if collision_count > 0:
            self.collision_counts += collision_count
            self.logger.info(f"Step collision count: {collision_count}")
            self.logger.info(f"Total collisions so far: {self.collision_counts}")

        # 환경 스텝 진행
        for _ in range(self.env_params.sims_per_step):
            self._apply_rl_actions(rl_actions)
            self.k.simulation.simulation_step()

        states = self.get_state()
        rewards = self.compute_reward(rl_actions, collisions=collision_count, colliding_vehicles=colliding_vehicles)
        dones = super()._check_done()
        
        # RL 차량별 충돌 횟수 계산
        rl_collision_counts = {rl_id: 0 for rl_id in self.k.vehicle.get_rl_ids()}
        for i in range(0, len(colliding_vehicles), 2):
            veh1, veh2 = colliding_vehicles[i:i+2]
            if veh1 in rl_collision_counts:
                rl_collision_counts[veh1] += 1
            if veh2 in rl_collision_counts:
                rl_collision_counts[veh2] += 1
        
        # 정보 업데이트
        infos = {}
        for rl_id in states.keys():
            infos[rl_id] = {
                'total_collision_count': self.collision_counts,
                'new_collisions': collision_count,
                'vehicle_collision_count': rl_collision_counts[rl_id],  # 개별 차량의 충돌 수
                'colliding_vehicles': colliding_vehicles
            }

        return states, rewards, dones, infos

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
        dict
            각 에이전트의 observation이 포함된 딕셔너리
        """
        self.leader = []
        self.follower = []
        obs = {}

        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_pos = self.k.vehicle.get_x_by_id(rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            
            # 교차로 감지 로직 개선
            current_edge = self.k.vehicle.get_edge(rl_id)
            if ':center' in str(current_edge):  # 교차로 진입 확인
                pos = self.k.vehicle.get_position(rl_id)
                self.logger.info(
                    f"Vehicle {rl_id} at intersection - "
                    f"Position: ({pos[0]:.2f}, {pos[1]:.2f}), "
                    f"Speed: {this_speed:.2f} m/s, "
                    f"Edge: {current_edge}"
                )

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

        # 항상 딕셔너리 형태로 반환
        return obs

    def _apply_rl_actions(self, rl_actions):
        """Apply acceleration actions from RL agents."""
        if rl_actions:
            for rl_id, acceleration in rl_actions.items():
                self.k.vehicle.apply_acceleration(rl_id, acceleration)

    def compute_reward(self, rl_actions, **kwargs):
        """Calculate reward focusing on target velocity in intersections."""
        if rl_actions is None:
            return {}

        # 기본 파라미터 설정
        rewards_dict = {}
        target_vel = self.env_params.additional_params['target_velocity']
        
        # 충돌 정보 획득
        rl_collision_counts = {rl_id: 0 for rl_id in rl_actions.keys()}
        colliding_vehicles = kwargs.get('colliding_vehicles', [])
        
        # 충돌 카운트 업데이트
        for i in range(0, len(colliding_vehicles), 2):
            veh1, veh2 = colliding_vehicles[i:i+2]
            if veh1 in rl_collision_counts:
                rl_collision_counts[veh1] += 1
            if veh2 in rl_collision_counts:
                rl_collision_counts[veh2] += 1

        for rl_id in rl_actions.keys():
            current_edge = self.k.vehicle.get_edge(rl_id)
            current_speed = self.k.vehicle.get_speed(rl_id)
            
            # 1. 교차로 구간 reward 계산
            intersection_reward = 0
            if ':center' in str(current_edge):  # 교차로 구간
                # target_velocity와의 차이에 기반한 reward
                speed_diff = abs(current_speed - target_vel)
                intersection_reward = 1.0 * (1 - speed_diff / target_vel)
                
                # 로깅
                self.logger.info(
                    f"Intersection - Vehicle {rl_id}: "
                    f"Speed={current_speed:.2f}, "
                    f"Target={target_vel:.2f}, "
                    f"Reward={intersection_reward:.2f}"
                )
            else:
                # 일반 도로 구간에서는 0 reward
                intersection_reward = 0
            
            # 2. 충돌 패널티 계산
            collision_penalty = self.env_params.additional_params.get('collision_penalty', 10)
            collision_reward = -rl_collision_counts[rl_id] * collision_penalty
            
            # 3. 최종 reward 계산 (교차로 reward 70% + 충돌 패널티 30%)
            final_reward = 0.7 * intersection_reward + 0.3 * collision_reward
            
            rewards_dict[rl_id] = final_reward

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