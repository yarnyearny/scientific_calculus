"""最短路と最小全域木のプリミティブ。"""

from __future__ import annotations

import heapq
from math import inf
from typing import Iterable

from .model import Edge, FaceGraph, edge_key


def dijkstra(
    graph: FaceGraph, source: str
) -> tuple[dict[str, float], dict[str, str]]:
    """単一始点から到達可能な全頂点への距離と直前頂点を返す。"""
    graph.centroid(source)
    distances = {source: 0.0}
    previous: dict[str, str] = {}
    queue: list[tuple[float, str]] = [(0.0, source)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor in graph.neighbors(node):
            candidate = distance + graph.weight(node, neighbor)
            current = distances.get(neighbor, inf)
            if candidate < current:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    return distances, previous


def reconstruct_path(
    previous: dict[str, str], source: str, target: str
) -> tuple[str, ...]:
    """Dijkstra の直前頂点表から経路を復元する。"""
    path = [target]
    while path[-1] != source:
        try:
            path.append(previous[path[-1]])
        except KeyError as error:
            raise ValueError(f"{source!r} から {target!r} へ到達できません") from error
    path.reverse()
    return tuple(path)


def shortest_path_to_any(
    graph: FaceGraph, source: str, targets: Iterable[str]
) -> tuple[str, ...]:
    """始点から対象集合の最寄り頂点までの決定的な最短路を返す。"""
    target_set = frozenset(targets)
    if not target_set:
        raise ValueError("最短路の対象集合は空にできません")
    unknown = target_set - graph.nodes
    if unknown:
        raise ValueError(f"最短路の対象に未知の面があります: {sorted(unknown)!r}")
    if source in target_set:
        return (source,)

    distances = {source: 0.0}
    previous: dict[str, str] = {}
    queue: list[tuple[float, str]] = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node in target_set:
            return reconstruct_path(previous, source, node)
        for neighbor in graph.neighbors(node):
            candidate = distance + graph.weight(node, neighbor)
            current = distances.get(neighbor, inf)
            if candidate < current:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    raise ValueError(f"{source!r} から対象集合へ到達できません")


def path_edges(path: Iterable[str]) -> tuple[Edge, ...]:
    nodes = tuple(path)
    return tuple(edge_key(left, right) for left, right in zip(nodes, nodes[1:]))


class DisjointSet:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in self.parent}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def minimum_spanning_edges(
    graph: FaceGraph,
    candidates: Iterable[Edge],
    required_nodes: Iterable[str],
) -> frozenset[Edge]:
    """候補辺上で required_nodes を含む最小全域木を返す。"""
    candidate_set = {edge_key(*edge) for edge in candidates}
    required_set = set(required_nodes)
    nodes = set(required_set)
    for left, right in candidate_set:
        nodes.add(left)
        nodes.add(right)
    if not nodes:
        return frozenset()

    disjoint = DisjointSet(nodes)
    selected: set[Edge] = set()
    ordered = sorted(
        candidate_set, key=lambda edge: (graph.weight(*edge), edge[0], edge[1])
    )
    for edge in ordered:
        if disjoint.union(*edge):
            selected.add(edge)

    roots = {disjoint.find(node) for node in required_set}
    if len(roots) > 1:
        raise ValueError("候補辺では必要な面を連結できません")
    return frozenset(selected)


def prune_nonterminal_leaves(
    edges: Iterable[Edge], terminals: Iterable[str]
) -> frozenset[Edge]:
    """木から終端ではない葉を再帰的に除去する。"""
    selected = {edge_key(*edge) for edge in edges}
    terminal_set = set(terminals)
    adjacency: dict[str, set[str]] = {}
    for left, right in selected:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    queue = [
        node
        for node, neighbors in adjacency.items()
        if len(neighbors) <= 1 and node not in terminal_set
    ]
    heapq.heapify(queue)
    while queue:
        leaf = heapq.heappop(queue)
        neighbors = adjacency.get(leaf)
        if neighbors is None or len(neighbors) != 1 or leaf in terminal_set:
            continue
        neighbor = neighbors.pop()
        selected.remove(edge_key(leaf, neighbor))
        adjacency[neighbor].remove(leaf)
        del adjacency[leaf]
        if len(adjacency[neighbor]) == 1 and neighbor not in terminal_set:
            heapq.heappush(queue, neighbor)
    return frozenset(selected)


def normalize_tree(
    graph: FaceGraph, edges: Iterable[Edge], terminals: Iterable[str]
) -> frozenset[Edge]:
    """候補辺集合を MST 化し、不要な非終端葉を除く。"""
    terminal_set = frozenset(terminals)
    tree = minimum_spanning_edges(graph, edges, terminal_set)
    return prune_nonterminal_leaves(tree, terminal_set)
