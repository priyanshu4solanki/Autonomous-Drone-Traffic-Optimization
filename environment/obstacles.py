from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: int
    y: int

@dataclass(frozen=True)
class Rectangle:
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
