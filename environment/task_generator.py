import random

from .obstacles import Point


class TaskGenerator:
    def __init__(self, city, seed=None):
        self.city, self.rng = city, random.Random(seed)

    def generate_tasks(self, num_tasks, tasks=None):
        if num_tasks <= 0:
            raise ValueError("num_tasks must be greater than zero")

        if tasks is not None:
            return self.validate_tasks(num_tasks, tasks)

        free_cells = self.city.all_free_cells()
        if len(free_cells) < num_tasks * 2:
            raise ValueError(
                f"Cannot create {num_tasks} distinct drone tasks: "
                f"city has only {len(free_cells)} free cells"
            )

        available_sources = self.city.warehouses or free_cells
        if len(available_sources) < num_tasks:
            available_sources = free_cells
        sources = self.rng.sample(available_sources, num_tasks)
        source_set = set(sources)

        available_destinations = [
            point for point in self.city.delivery_points
            if point not in source_set
        ]
        if len(available_destinations) < num_tasks:
            available_destinations = [
                point for point in free_cells
                if point not in source_set
            ]
        destinations = self.rng.sample(available_destinations, num_tasks)

        return list(zip(sources, destinations))

    def validate_tasks(self, num_tasks, tasks):
        if len(tasks) != num_tasks:
            raise ValueError(
                f"Expected {num_tasks} delivery tasks, received {len(tasks)}"
            )

        normalized = []
        for index, (source, destination) in enumerate(tasks, start=1):
            source = self._as_point(source)
            destination = self._as_point(destination)
            for label, point in (("source", source), ("destination", destination)):
                if not self.city.inside_city(point.x, point.y) or self.city.is_blocked(point.x, point.y):
                    raise ValueError(f"Task {index} {label} must be a free city cell")
            if source == destination:
                raise ValueError(f"Task {index} source and destination must differ")
            normalized.append((source, destination))

        sources = [source for source, _ in normalized]
        destinations = [destination for _, destination in normalized]
        if len(set(sources)) != num_tasks:
            raise ValueError("Each drone must have a distinct starting cell")
        if len(set(destinations)) != num_tasks:
            raise ValueError("Each drone must have a distinct destination cell")
        if set(sources) & set(destinations):
            raise ValueError("Starting cells and destination cells must be distinct")

        return normalized

    @staticmethod
    def _as_point(value):
        if isinstance(value, Point):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return Point(int(value[0]), int(value[1]))
        raise ValueError("Task locations must be Points or (x, y) pairs")