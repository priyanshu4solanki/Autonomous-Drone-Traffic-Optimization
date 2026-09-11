import random
from .obstacles import Point, Rectangle


class City:
    def __init__(
        self,
        width=25,
        height=25,
        num_buildings=12,
        num_no_fly_zones=5,
        num_warehouses=20,
        num_delivery_points=100,
        num_charging_stations=5,
        seed=None,
    ):
        self.width, self.height, self.rng = width, height, random.Random(seed)
        self.buildings, self.no_fly_zones, self.warehouses, self.delivery_points, self.charging_stations = [], [], [], [], []

        self._generate_rectangles(num_buildings, is_no_fly=False)
        self._generate_rectangles(num_no_fly_zones, is_no_fly=True)

        self._generate_points(self.warehouses, num_warehouses)
        self._generate_points(self.delivery_points, num_delivery_points)
        self._generate_points(self.charging_stations, num_charging_stations)

    def _generate_rectangles(self, count, is_no_fly):
        target = self.no_fly_zones if is_no_fly else self.buildings
        for _ in range(count * 300):
            if len(target) >= count:
                break
            w = self.rng.randint(1, min(4, max(1, self.width // 4)))
            h = self.rng.randint(1, min(4, max(1, self.height // 4)))
            x1, y1 = self.rng.randint(0, self.width - w), self.rng.randint(0, self.height - h)
            rect = Rectangle(x1, y1, x1 + w - 1, y1 + h - 1)
            
            if not any(self._rect_overlap(rect, r) for r in self.buildings + self.no_fly_zones):
                target.append(rect)

    @staticmethod
    def _rect_overlap(a, b):
        return not (a.x2 < b.x1 or a.x1 > b.x2 or a.y2 < b.y1 or a.y1 > b.y2)

    def _generate_points(self, target, count):
        free = self.all_free_cells()
        self.rng.shuffle(free)
        existing = self.warehouses + self.delivery_points + self.charging_stations
        
        for pt in free:
            if len(target) >= count:
                break
            if pt not in existing:
                target.append(pt)
                existing.append(pt)

    def is_building(self, x, y):
        return any(r.contains(x, y) for r in self.buildings)

    def is_no_fly_zone(self, x, y):
        return any(r.contains(x, y) for r in self.no_fly_zones)

    def is_blocked(self, x, y):
        return self.is_building(x, y) or self.is_no_fly_zone(x, y)

    def inside_city(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_charging_station(self, x, y):
        return Point(x, y) in self.charging_stations

    def is_service_point(self, x, y):
        p = Point(x, y)
        return p in self.warehouses or p in self.delivery_points or p in self.charging_stations

    def all_free_cells(self):
        return [Point(x, y) for x in range(self.width) for y in range(self.height) if not self.is_blocked(x, y)]