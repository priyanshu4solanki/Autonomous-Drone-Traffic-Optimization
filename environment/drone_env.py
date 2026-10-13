import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .city import City
from .drone import Drone
from .task_generator import TaskGenerator
from .observations import ObservationBuilder
from .movement import execute_action, legal_action_mask
from .collisions import detect_collision_pairs
from .rewards import RewardCalculator


class DroneTrafficEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        
        self.config = {
            "width": 25, "height": 25, "num_buildings": 10, "num_no_fly_zones": 5,
            "num_warehouses": 20, "num_delivery_points": 10, "num_charging_stations": 5,
            "num_drones": 5, "battery": 100.0, "movement_battery_cost": 1.0,
            "wait_battery_cost": 0.1, "recharge_amount": 15.0, "observation_radius": 4,
            "max_steps": 300, "delivery_reward": 100.0, "collision_penalty": -80.0,
            "illegal_action_penalty": -2.0, "battery_failure_penalty": -40.0,
            "step_penalty": -0.05, "progress_reward": 2.0, "wait_penalty": -0.10,
        }
        if config:
            self.config.update(config)

        self.num_drones = self.config["num_drones"]
        self.action_space = spaces.MultiDiscrete([6] * self.num_drones)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_drones, 18), dtype=np.float32)
        self.global_state_dim = self.num_drones * 7 + 3

        self.city = None
        self.drones = []
        self.observations = None
        self.step_count = 0
        self.total_deliveries = 0
        self.total_collision_events = 0
        self.total_collision_involvements = 0
        self.reward_calculator = RewardCalculator(self.config)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        seed_val = int(seed) if seed is not None else None

        self.city = City(
            width=self.config["width"], height=self.config["height"],
            num_buildings=self.config["num_buildings"], num_no_fly_zones=self.config["num_no_fly_zones"],
            num_warehouses=self.config["num_warehouses"], num_delivery_points=self.config["num_delivery_points"],
            num_charging_stations=self.config["num_charging_stations"], seed=seed_val,
        )

        task_gen = TaskGenerator(self.city, seed=None if seed_val is None else seed_val + 1)
        tasks = task_gen.generate_tasks(
            self.num_drones,
            tasks=None if options is None else options.get("tasks"),
        )

        self.drones = [
            Drone(drone_id=i, position=src, destination=dst, battery=self.config["battery"], initial_battery=self.config["battery"])
            for i, (src, dst) in enumerate(tasks)
        ]

        self.observations = ObservationBuilder(self.city, self.drones, self.config["observation_radius"])
        self.step_count = self.total_deliveries = 0
        self.total_collision_events = self.total_collision_involvements = 0

        return self.get_local_observations(), self._info()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_drones)
        prev_pos = [d.position for d in self.drones]
        prev_dist = [self.reward_calculator.distance(d.position, d.destination) for d in self.drones]
        active_before_step = [d.active for d in self.drones]
        action_masks = self.get_action_masks()

        events = []
        for i, (d, act) in enumerate(zip(self.drones, actions)):
            res = execute_action(
                d,
                int(act),
                self.city,
                self.config,
                action_mask=action_masks[i],
            )
            events.append((res, int(act) == 4 and not res["illegal"]))

        new_pos = [d.position for d in self.drones]
        collision_pairs = detect_collision_pairs(
            prev_pos,
            new_pos,
            [True] * self.num_drones,
        )
        collision_flags = [False] * self.num_drones
        for i, j in collision_pairs:
            collision_flags[i] = collision_flags[j] = True

        conflicting_targets = {}
        blocked_targets = set()
        rejected_moves = set()
        for i, j in collision_pairs:
            both_moving_to_same_cell = (
                events[i][0]["moved"]
                and events[j][0]["moved"]
                and new_pos[i] == new_pos[j]
            )
            if both_moving_to_same_cell:
                conflicting_targets.setdefault(new_pos[i], set()).update((i, j))
            else:
                if events[i][0]["moved"]:
                    rejected_moves.add(i)
                if events[j][0]["moved"]:
                    rejected_moves.add(j)
                if new_pos[i] == new_pos[j]:
                    blocked_targets.add(new_pos[i])

        for target, contenders in conflicting_targets.items():
            if target in blocked_targets:
                rejected_moves.update(contenders)
                continue
            winner = min(
                contenders,
                key=lambda index: (self.drones[index].drone_id - self.step_count) % self.num_drones,
            )
            rejected_moves.update(contenders - {winner})

        # Cancel rejected moves and restore the battery they would have used.
        for i in rejected_moves:
            drone = self.drones[i]
            candidate = drone.position
            drone.position = prev_pos[i]
            drone.distance_travelled -= self.reward_calculator.distance(prev_pos[i], candidate)
            drone.trajectory.pop()
            drone.battery = min(
                drone.initial_battery,
                drone.battery + self.config["movement_battery_cost"],
            )

        collision_flags = [i in rejected_moves for i in range(self.num_drones)]
        self.total_collision_events += len(collision_pairs)
        self.total_collision_involvements += sum(
            collided and active_before_step[i]
            for i, collided in enumerate(collision_flags)
        )
        rewards = np.zeros(self.num_drones, dtype=np.float32)

        for i, d in enumerate(self.drones):
            res, waited = events[i]
            if collision_flags[i] and active_before_step[i]: d.collisions += 1

            delivered_now = False
            if d.active and d.position == d.destination:
                d.delivered, d.active, delivered_now = True, False, True
                self.total_deliveries += 1

            battery_failure = d.active and d.battery <= 0 and d.position != d.destination
            if battery_failure: d.active = False

            rewards[i] = self.reward_calculator.calculate(
                d, prev_dist[i],
                collision=collision_flags[i] and active_before_step[i],
                illegal=res["illegal"],
                battery_failure=battery_failure, delivered_now=delivered_now, waited=waited
            )
            d.steps_taken += 1

        self.step_count += 1
        terminated = all(not d.active for d in self.drones)
        truncated = self.step_count >= self.config["max_steps"]

        info = self._info()
        info.update({
            "deliveries": self.total_deliveries,
            "battery_remaining": float(np.mean([d.battery for d in self.drones])),
            "agent_rewards": rewards.copy(),
        })

        return (
            self.get_local_observations(),
            float(np.sum(rewards)),
            terminated,
            truncated,
            info,
        )

    def get_local_observations(self):
        return self.observations.all_local()

    def get_local_maps(self):
        return self.observations.all_local_maps()

    def get_global_state(self):
        return self.observations.global_state()

    def get_action_masks(self):
        return np.asarray([
            legal_action_mask(d, self.city, self.config, self.drones)
            for d in self.drones
        ], dtype=bool)

    def _info(self):
        return {
            "deliveries": self.total_deliveries,
            "collisions": self.total_collision_events,
            "collision_involvements": self.total_collision_involvements,
            "no_fly_violations": sum(d.no_fly_violations for d in self.drones),
        }

    def render(self):
        print(f"step={self.step_count} deliveries={self.total_deliveries}/{self.num_drones} collisions={self._info()['collisions']}")

    def close(self):
        pass