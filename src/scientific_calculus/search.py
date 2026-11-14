"""スタート順列に対する再始動付き局所探索。"""

from __future__ import annotations

import random
from collections.abc import Callable

from .model import TreeSolution

Evaluator = Callable[[tuple[str, ...]], TreeSolution]


def solution_key(solution: TreeSolution) -> tuple[float, tuple[tuple[str, str], ...]]:
    return solution.cost, tuple(sorted(solution.edges))


def _mutate(order: tuple[str, ...], generator: random.Random) -> tuple[str, ...]:
    if len(order) < 2:
        return order
    items = list(order)
    left, right = sorted(generator.sample(range(len(items)), 2))
    operation = generator.randrange(3)
    if operation == 0:
        items[left], items[right] = items[right], items[left]
    elif operation == 1:
        item = items.pop(right)
        items.insert(left, item)
    else:
        items[left : right + 1] = reversed(items[left : right + 1])
    return tuple(items)


def optimize_order(
    starts: tuple[str, ...],
    evaluator: Evaluator,
    *,
    random_restarts: int,
    local_iterations: int,
    seed: int,
) -> TreeSolution:
    """複数のランダム初期順列を山登り法で改善する。"""
    if random_restarts < 0 or local_iterations < 0:
        raise ValueError("探索回数は0以上である必要があります")

    generator = random.Random(seed)
    canonical = tuple(sorted(set(starts)))
    cache: dict[tuple[str, ...], TreeSolution] = {}

    def evaluate(order: tuple[str, ...]) -> TreeSolution:
        if order not in cache:
            cache[order] = evaluator(order)
        return cache[order]

    best = evaluate(canonical)
    restart_count = max(1, random_restarts)
    for restart in range(restart_count):
        if restart == 0:
            current_order = canonical
        else:
            shuffled = list(canonical)
            generator.shuffle(shuffled)
            current_order = tuple(shuffled)
        current = evaluate(current_order)

        for _ in range(local_iterations):
            candidate_order = _mutate(current_order, generator)
            candidate = evaluate(candidate_order)
            if solution_key(candidate) < solution_key(current):
                current_order = candidate_order
                current = candidate
            if solution_key(current) < solution_key(best):
                best = current
    return best
