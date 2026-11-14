"""問題領域のモデルと不変条件。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import fsum, isfinite
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

Edge = tuple[str, str]
Point = tuple[float, float]


def edge_key(left: str, right: str) -> Edge:
    """無向辺を安定した順序に正規化する。"""
    if left == right:
        raise ValueError(f"自己辺は指定できません: {left!r}")
    return (left, right) if left < right else (right, left)


class FaceGraph:
    """面の重心と無向隣接関係を保持する不変グラフ。"""

    def __init__(
        self,
        centroids: Mapping[str, Sequence[float]],
        adjacencies: Iterable[Sequence[str]],
    ) -> None:
        if not centroids:
            raise ValueError("faces は1件以上必要です")

        points: dict[str, Point] = {}
        for face_id, coordinates in centroids.items():
            if not isinstance(face_id, str) or not face_id:
                raise ValueError("面IDは空でない文字列である必要があります")
            if len(coordinates) != 2:
                raise ValueError(f"重心は2次元座標である必要があります: {face_id!r}")
            x, y = coordinates
            if isinstance(x, bool) or isinstance(y, bool):
                raise ValueError(f"重心座標は数値である必要があります: {face_id!r}")
            try:
                point = (float(x), float(y))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"重心座標は数値である必要があります: {face_id!r}"
                ) from error
            if not all(isfinite(value) for value in point):
                raise ValueError(f"重心座標は有限値である必要があります: {face_id!r}")
            points[face_id] = point

        neighbors: dict[str, set[str]] = {face_id: set() for face_id in points}
        edges: set[Edge] = set()
        for adjacency in adjacencies:
            if len(adjacency) != 2:
                raise ValueError("adjacencies の各要素は2面の組である必要があります")
            left, right = adjacency
            if left not in points or right not in points:
                raise ValueError(f"隣接関係が未知の面を参照しています: {adjacency!r}")
            edge = edge_key(left, right)
            edges.add(edge)
            neighbors[left].add(right)
            neighbors[right].add(left)

        self._centroids = MappingProxyType(points)
        self._neighbors = MappingProxyType(
            {face_id: tuple(sorted(items)) for face_id, items in neighbors.items()}
        )
        self._edges = frozenset(edges)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self._centroids)

    @property
    def edges(self) -> frozenset[Edge]:
        return self._edges

    def centroid(self, face_id: str) -> Point:
        try:
            return self._centroids[face_id]
        except KeyError as error:
            raise ValueError(f"未知の面です: {face_id!r}") from error

    def neighbors(self, face_id: str) -> tuple[str, ...]:
        try:
            return self._neighbors[face_id]
        except KeyError as error:
            raise ValueError(f"未知の面です: {face_id!r}") from error

    def weight(self, left: str, right: str) -> float:
        edge = edge_key(left, right)
        if edge not in self._edges:
            raise ValueError(f"隣接していない面の組です: {edge!r}")
        left_point = self._centroids[left]
        right_point = self._centroids[right]
        return abs(left_point[0] - right_point[0]) + abs(
            left_point[1] - right_point[1]
        )

    def validate_terminals(self, goal: str, starts: Iterable[str]) -> tuple[str, ...]:
        if goal not in self._centroids:
            raise ValueError(f"ゴール面が存在しません: {goal!r}")
        normalized = tuple(sorted(set(starts)))
        if not normalized:
            raise ValueError("スタート面は1件以上必要です")
        unknown = [face_id for face_id in normalized if face_id not in self._centroids]
        if unknown:
            raise ValueError(f"存在しないスタート面です: {unknown!r}")

        reachable = {goal}
        queue = deque([goal])
        while queue:
            node = queue.popleft()
            for neighbor in self._neighbors[node]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        unreachable = [face_id for face_id in normalized if face_id not in reachable]
        if unreachable:
            raise ValueError(f"ゴールへ到達できないスタート面です: {unreachable!r}")
        return normalized


@dataclass(frozen=True, slots=True)
class TreeSolution:
    """ゴールを根とする共有経路木。"""

    goal: str
    starts: tuple[str, ...]
    edges: frozenset[Edge]
    nodes: frozenset[str]
    cost: float

    @classmethod
    def create(
        cls,
        graph: FaceGraph,
        goal: str,
        starts: Iterable[str],
        edges: Iterable[Edge],
    ) -> TreeSolution:
        normalized_starts = graph.validate_terminals(goal, starts)
        normalized_edges = frozenset(edge_key(*edge) for edge in edges)
        unknown_edges = normalized_edges - graph.edges
        if unknown_edges:
            raise ValueError(f"解がグラフにない辺を含みます: {sorted(unknown_edges)!r}")

        nodes = {goal, *normalized_starts}
        for left, right in normalized_edges:
            nodes.add(left)
            nodes.add(right)

        adjacency: dict[str, set[str]] = {node: set() for node in nodes}
        for left, right in normalized_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        reachable = {goal}
        queue = deque([goal])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        if reachable != nodes:
            raise ValueError("解は連結である必要があります")
        if len(normalized_edges) != len(nodes) - 1:
            raise ValueError("解は閉路を含まない木である必要があります")

        cost = fsum(graph.weight(*edge) for edge in normalized_edges)
        return cls(
            goal=goal,
            starts=normalized_starts,
            edges=normalized_edges,
            nodes=frozenset(nodes),
            cost=cost,
        )

    def oriented_edges(self) -> tuple[Edge, ...]:
        """辺を `(親, 子)` としてゴールから幅優先順に返す。"""
        adjacency: dict[str, list[str]] = {node: [] for node in self.nodes}
        for left, right in self.edges:
            adjacency[left].append(right)
            adjacency[right].append(left)
        for neighbors in adjacency.values():
            neighbors.sort()

        result: list[Edge] = []
        visited = {self.goal}
        queue = deque([self.goal])
        while queue:
            parent = queue.popleft()
            for child in adjacency[parent]:
                if child not in visited:
                    visited.add(child)
                    result.append((parent, child))
                    queue.append(child)
        return tuple(result)
