import random


class TaskGenerator:
    def __init__(self, city, seed=None):
        self.city, self.rng = city, random.Random(seed)

    def generate_tasks(self, num_tasks):
        sources = self.city.warehouses or self.city.all_free_cells()
        destinations = self.city.delivery_points or self.city.all_free_cells()

        tasks = []
        for _ in range(num_tasks):
            src = self.rng.choice(sources)
            dst = self.rng.choice([p for p in destinations if p != src] or destinations)
            tasks.append((src, dst))
        return tasks