"""共有経路木の初期解を構築するアルゴリズム。"""

from __future__ import annotations

from .model import Edge, FaceGraph, TreeSolution
from .paths import (
    DisjointSet,
    dijkstra,
    normalize_tree,
    path_edges,
    reconstruct_path,
    shortest_path_to_any,
)


def sequential_attachment(
    graph: FaceGraph,
    goal: str,
    starts: tuple[str, ...],
    order: tuple[str, ...],
) -> TreeSolution:
    """各スタートを指定順で現在の木へ最短接続する。"""
    normalized_starts = graph.validate_terminals(goal, starts)
    expected = set(normalized_starts)
    if len(order) != len(normalized_starts) or set(order) != expected:
        raise ValueError("order は各スタートをちょうど一度ずつ含む必要があります")

    tree_nodes = {goal}
    candidates: set[Edge] = set()
    for start in order:
        path = shortest_path_to_any(graph, start, tree_nodes)
        candidates.update(path_edges(path))
        tree_nodes.update(path)

    terminals = {goal, *normalized_starts}
    tree = normalize_tree(graph, candidates, terminals)
    return TreeSolution.create(graph, goal, normalized_starts, tree)


def metric_closure_tree(
    graph: FaceGraph, goal: str, starts: tuple[str, ...]
) -> TreeSolution:
    """終端距離閉包の MST を元グラフへ展開して近似木を作る。"""
    normalized_starts = graph.validate_terminals(goal, starts)
    terminals = tuple(sorted({goal, *normalized_starts}))
    if len(terminals) == 1:
        return TreeSolution.create(graph, goal, normalized_starts, ())

    shortest: dict[str, tuple[dict[str, float], dict[str, str]]] = {
        terminal: dijkstra(graph, terminal) for terminal in terminals
    }
    closure_edges: list[tuple[float, str, str]] = []
    for index, left in enumerate(terminals):
        distances, _ = shortest[left]
        for right in terminals[index + 1 :]:
            closure_edges.append((distances[right], left, right))
    closure_edges.sort()

    disjoint = DisjointSet(terminals)
    selected_closure: list[tuple[str, str]] = []
    for _, left, right in closure_edges:
        if disjoint.union(left, right):
            selected_closure.append((left, right))
            if len(selected_closure) == len(terminals) - 1:
                break

    candidates: set[Edge] = set()
    for source, target in selected_closure:
        _, previous = shortest[source]
        path = reconstruct_path(previous, source, target)
        candidates.update(path_edges(path))

    tree = normalize_tree(graph, candidates, terminals)
    return TreeSolution.create(graph, goal, normalized_starts, tree)
