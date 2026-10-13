import math
import numpy as np


class ObservationBuilder:
    MAP_CHANNELS = (
        "buildings",
        "no_fly_zones",
        "charging_stations",
        "other_drones",
        "self",
        "destination",
    )

    def __init__(self, city, drones, observation_radius=4):
        self.city, self.drones, self.radius = city, drones, observation_radius

    @property
    def obs_dim(self):
        return 18

    def local_observation(self, drone):
        x, y = drone.position.x, drone.position.y
        dx, dy = drone.destination.x - x, drone.destination.y - y
        ndx, ndy, ndist = self._nearest_drone(drone)

        values = [
            self._relative(dx),
            self._relative(dy),
            drone.battery / max(drone.initial_battery, 1e-6),
            float(self._blocked(x, y + 1)),
            float(self._blocked(x, y - 1)),
            float(self._blocked(x - 1, y)),
            float(self._blocked(x + 1, y)),
            float(self._drone_at(x, y + 1, drone)),
            float(self._drone_at(x, y - 1, drone)),
            float(self._drone_at(x - 1, y, drone)),
            float(self._drone_at(x + 1, y, drone)),
            float(np.clip(ndist / max(self.radius, 1), 0.0, 1.0)),
            self._relative(ndx),
            self._relative(ndy),
            self._traffic_density(drone),
            float(self.city.is_charging_station(x, y)),
            float(drone.delivered),
            float(drone.active),
        ]
        return np.asarray(values, dtype=np.float32)

    def all_local(self):
        return np.asarray([self.local_observation(d) for d in self.drones], dtype=np.float32)

    def local_map(self, drone):
        size = 2 * self.radius + 1
        local_map = np.zeros((len(self.MAP_CHANNELS), size, size), dtype=np.float32)
        center_x, center_y = drone.position.x, drone.position.y

        for row, dy in enumerate(range(-self.radius, self.radius + 1)):
            for col, dx in enumerate(range(-self.radius, self.radius + 1)):
                x, y = center_x + dx, center_y + dy
                if not self.city.inside_city(x, y):
                    local_map[0, row, col] = 1.0
                    continue

                local_map[0, row, col] = float(self.city.is_building(x, y))
                local_map[1, row, col] = float(self.city.is_no_fly_zone(x, y))
                local_map[2, row, col] = float(self.city.is_charging_station(x, y))
                local_map[3, row, col] = float(self._drone_at(x, y, drone))
                local_map[4, row, col] = float(x == center_x and y == center_y)
                local_map[5, row, col] = float(
                    x == drone.destination.x and y == drone.destination.y
                )

        return local_map

    def all_local_maps(self):
        return np.asarray([self.local_map(d) for d in self.drones], dtype=np.float32)

    def global_state(self):
        state = [
            val
            for d in self.drones
            for val in (
                self._norm(d.position.x, self.city.width),
                self._norm(d.position.y, self.city.height),
                self._norm(d.destination.x, self.city.width),
                self._norm(d.destination.y, self.city.height),
                d.battery / max(d.initial_battery, 1e-6),
                float(d.active),
                float(d.delivered),
            )
        ]
        state.extend([
            self._traffic_density_all(),
            len(self.city.buildings) / max(self.city.width, 1),
            len(self.city.no_fly_zones) / max(self.city.width, 1),
        ])
        return np.asarray(state, dtype=np.float32)

    def _nearest_drone(self, drone):
        active_others = [o for o in self.drones if o is not drone and o.active]
        if not active_others:
            return 0.0, 0.0, float(self.radius)

        distances = [
            (o.position.x - drone.position.x, o.position.y - drone.position.y, math.hypot(o.position.x - drone.position.x, o.position.y - drone.position.y))
            for o in active_others
        ]
        return min(distances, key=lambda item: item[2])

    def _drone_at(self, x, y, drone):
        return any(
            d is not drone and d.position.x == x and d.position.y == y
            for d in self.drones
        )

    def _traffic_density(self, drone):
        active_others = [d for d in self.drones if d is not drone and d.active]
        if not active_others:
            return 0.0
        r = self.radius
        x, y = drone.position.x, drone.position.y
        close_count = sum(1 for d in active_others if abs(d.position.x - x) <= r and abs(d.position.y - y) <= r)
        return close_count / len(active_others)

    def _traffic_density_all(self):
        active = [d for d in self.drones if d.active]
        if len(active) <= 1:
            return 0.0
        
        total_pairs, close_pairs = 0, 0
        r = self.radius
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                total_pairs += 1
                if abs(active[i].position.x - active[j].position.x) <= r and abs(active[i].position.y - active[j].position.y) <= r:
                    close_pairs += 1
        return close_pairs / max(total_pairs, 1)

    def _blocked(self, x, y):
        return not self.city.inside_city(x, y) or self.city.is_blocked(x, y)

    def _relative(self, value):
        return float(np.clip(value / max(self.city.width, self.city.height), -1.0, 1.0))

    @staticmethod
    def _norm(value, size):
        return float(value / max(size - 1, 1))