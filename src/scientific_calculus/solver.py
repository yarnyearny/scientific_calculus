"""候補生成と局所探索を統合する公開ソルバー。"""

from __future__ import annotations

from dataclasses import dataclass

from .construction import metric_closure_tree, sequential_attachment
from .model import FaceGraph, TreeSolution
from .paths import dijkstra
from .search import optimize_order, solution_key


@dataclass(frozen=True, slots=True)
class SolverConfig:
    random_restarts: int = 16
    local_iterations: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("random_restarts", "local_iterations", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} は整数である必要があります")
        if self.random_restarts < 0 or self.local_iterations < 0:
            raise ValueError("探索回数は0以上である必要があります")


def solve(
    graph: FaceGraph,
    goal: str,
    starts: tuple[str, ...],
    config: SolverConfig | None = None,
) -> TreeSolution:
    """複数の近似構築法を比較し、最良の共有経路木を返す。"""
    settings = config or SolverConfig()
    normalized_starts = graph.validate_terminals(goal, starts)
    distances, _ = dijkstra(graph, goal)

    nearest_first = tuple(
        sorted(normalized_starts, key=lambda node: (distances[node], node))
    )
    farthest_first = tuple(
        sorted(normalized_starts, key=lambda node: (-distances[node], node))
    )

    def evaluate(order: tuple[str, ...]) -> TreeSolution:
        return sequential_attachment(
            graph, goal, normalized_starts, order
        )

    candidates = [
        metric_closure_tree(graph, goal, normalized_starts),
        evaluate(nearest_first),
        evaluate(farthest_first),
        optimize_order(
            normalized_starts,
            evaluate,
            random_restarts=settings.random_restarts,
            local_iterations=settings.local_iterations,
            seed=settings.seed,
        ),
    ]
    return min(candidates, key=solution_key)
