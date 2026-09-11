from dataclasses import dataclass, field
import math
from .obstacles import Point


@dataclass
class Drone:
    drone_id: int
    position: Point
    destination: Point
    battery: float
    initial_battery: float
    active: bool = True
    delivered: bool = False
    trajectory: list = field(default_factory=list)
    distance_travelled: float = 0.0
    steps_taken: int = 0
    collisions: int = 0
    no_fly_violations: int = 0

    def __post_init__(self):
        self.trajectory.append(self.position)

    def move_to(self, new_position):
        self.distance_travelled += math.hypot(new_position.x - self.position.x, new_position.y - self.position.y)
        self.position = new_position
        self.trajectory.append(new_position)

    def consume_battery(self, amount):
        self.battery = max(0.0, self.battery - amount)

    def recharge(self, amount):
        self.battery = min(self.initial_battery, self.battery + amount)