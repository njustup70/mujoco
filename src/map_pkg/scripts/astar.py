import heapq
from dataclasses import dataclass
from math import hypot, sqrt

import numpy as np

GridCell = tuple[int, int]
GrideCell = GridCell


@dataclass
class AstarConfig:
    cost_weight: float = 5.0
    lethal_cost: float = 100.0
    allow_diagonal: bool = True
    unknown_is_obstacle: bool = True


class AStarPlanner:
    def __init__(self, config: AstarConfig | None = None):
        self.config = config or AstarConfig()
        if self.config.lethal_cost <= 0.0:
            raise ValueError("lethal_cost must be greater than 0")
        if self.config.cost_weight < 0.0:
            raise ValueError("cost_weight must be non-negative")

    def plan(
        self,
        start: GridCell,
        goal: GridCell,
        cost_grid: np.ndarray,
    ) -> list[GridCell] | None:
        if cost_grid.ndim != 2:
            raise ValueError("cost_grid must be a 2D numpy array")

        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        if not self.in_bounds(start, cost_grid) or not self.in_bounds(goal, cost_grid):
            return None
        if not self.is_traversable(start, cost_grid):
            return None
        if not self.is_traversable(goal, cost_grid):
            return None

        counter = 0
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), counter, 0.0, start))

        came_from: dict[GridCell, GridCell | None] = {start: None}
        g_score: dict[GridCell, float] = {start: 0.0}
        closed_set: set[GridCell] = set()

        while open_set:
            _, _, current_g, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current, cost_grid):
                if neighbor in closed_set:
                    continue

                tentative_g_score = current_g + self.move_cost(
                    current, neighbor, cost_grid
                )

                if tentative_g_score >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + self.heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(
                    open_set, (f_score, counter, tentative_g_score, neighbor)
                )

        return None

    def heuristic(self, cell: GridCell, goal: GridCell) -> float:
        dx = abs(cell[1] - goal[1])
        dy = abs(cell[0] - goal[0])

        if self.config.allow_diagonal:
            straight = abs(dx - dy)
            diagonal = min(dx, dy)
            return straight + sqrt(2.0) * diagonal

        return float(dx + dy)

    def get_neighbors(
        self, cell: GridCell, cost_grid: np.ndarray
    ) -> list[GridCell]:
        row, col = cell
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.config.allow_diagonal:
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        neighbors = []
        for row_offset, col_offset in offsets:
            neighbor = (row + row_offset, col + col_offset)
            if not self.in_bounds(neighbor, cost_grid):
                continue
            if not self.is_traversable(neighbor, cost_grid):
                continue
            if self.config.allow_diagonal and row_offset != 0 and col_offset != 0:
                if not self.can_move_diagonal(cell, neighbor, cost_grid):
                    continue
            neighbors.append(neighbor)

        return neighbors

    def move_cost(
        self, current: GridCell, neighbor: GridCell, cost_grid: np.ndarray
    ) -> float:
        step_cost = hypot(neighbor[0] - current[0], neighbor[1] - current[1])
        normalized_cost = max(0.0, float(cost_grid[neighbor]) / self.config.lethal_cost)
        return step_cost * (1.0 + self.config.cost_weight * normalized_cost)

    def can_move_diagonal(
        self, current: GridCell, neighbor: GridCell, cost_grid: np.ndarray
    ) -> bool:
        current_row, current_col = current
        neighbor_row, neighbor_col = neighbor
        side_a = (current_row, neighbor_col)
        side_b = (neighbor_row, current_col)
        return self.is_traversable(side_a, cost_grid) and self.is_traversable(
            side_b, cost_grid
        )

    def in_bounds(self, cell: GridCell, cost_grid: np.ndarray) -> bool:
        row, col = cell
        height, width = cost_grid.shape
        return 0 <= row < height and 0 <= col < width

    def is_traversable(self, cell: GridCell, cost_grid: np.ndarray) -> bool:
        cost = float(cost_grid[cell])
        if self.config.unknown_is_obstacle and cost < 0:
            return False
        return cost < self.config.lethal_cost

    def reconstruct_path(
        self,
        came_from: dict[GridCell, GridCell | None],
        current: GridCell,
    ) -> list[GridCell]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def grid_to_world(
    cell: GridCell,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> tuple[float, float]:
    row, col = cell
    return (
        origin_x + (col + 0.5) * resolution,
        origin_y + (row + 0.5) * resolution,
    )


def world_to_grid(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> GridCell:
    col = int(np.floor((x - origin_x) / resolution))
    row = int(np.floor((y - origin_y) / resolution))
    return (row, col)
