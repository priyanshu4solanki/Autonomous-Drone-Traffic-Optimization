import math


class RewardCalculator:
    def __init__(self, config=None):
        cfg = config or {}
        self.delivery = cfg.get("delivery_reward", 100.0)
        self.collision = cfg.get("collision_penalty", -80.0)
        self.illegal = cfg.get("illegal_action_penalty", -2.0)
        self.battery_failure = cfg.get("battery_failure_penalty", -40.0)
        self.step = cfg.get("step_penalty", -0.05)
        self.progress = cfg.get("progress_reward", 2.0)
        self.wait_penalty = cfg.get("wait_penalty", -0.10)

    @staticmethod
    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def calculate(self, drone, prev_dist, collision=False, illegal=False, battery_failure=False, delivered_now=False, waited=False):
        reward = self.step + (prev_dist - self.distance(drone.position, drone.destination)) * self.progress

        if collision: reward += self.collision
        if illegal: reward += self.illegal
        if battery_failure: reward += self.battery_failure
        if delivered_now: reward += self.delivery
        if waited: reward += self.wait_penalty

        return float(reward)