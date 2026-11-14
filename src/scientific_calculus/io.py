"""JSON 表現とドメインモデルの変換。"""

from __future__ import annotations

from typing import Any

from .model import FaceGraph, TreeSolution
from .solver import SolverConfig


def parse_problem(
    payload: Any,
) -> tuple[FaceGraph, str, tuple[str, ...], SolverConfig]:
    if not isinstance(payload, dict):
        raise ValueError("入力JSONのルートはオブジェクトである必要があります")

    faces = payload.get("faces")
    if not isinstance(faces, list) or not faces:
        raise ValueError("faces は空でない配列である必要があります")
    centroids: dict[str, Any] = {}
    for face in faces:
        if not isinstance(face, dict):
            raise ValueError("faces の各要素はオブジェクトである必要があります")
        face_id = face.get("id")
        if not isinstance(face_id, str) or not face_id:
            raise ValueError("各面の id は空でない文字列である必要があります")
        if face_id in centroids:
            raise ValueError(f"面IDが重複しています: {face_id!r}")
        centroid = face.get("centroid")
        if not isinstance(centroid, (list, tuple)):
            raise ValueError(f"centroid は座標配列である必要があります: {face_id!r}")
        centroids[face_id] = centroid

    adjacencies = payload.get("adjacencies")
    if not isinstance(adjacencies, list):
        raise ValueError("adjacencies は配列である必要があります")
    for adjacency in adjacencies:
        if not isinstance(adjacency, (list, tuple)):
            raise ValueError("adjacencies の各要素は配列である必要があります")

    goal = payload.get("goal")
    if not isinstance(goal, str) or not goal:
        raise ValueError("goal は空でない文字列である必要があります")
    starts_value = payload.get("starts")
    if not isinstance(starts_value, list) or not all(
        isinstance(item, str) and item for item in starts_value
    ):
        raise ValueError("starts は空でない文字列の配列である必要があります")

    config_value = payload.get("config", {})
    if not isinstance(config_value, dict):
        raise ValueError("config はオブジェクトである必要があります")
    allowed_config = {"random_restarts", "local_iterations", "seed"}
    unknown_config = set(config_value) - allowed_config
    if unknown_config:
        raise ValueError(f"未知のconfig項目です: {sorted(unknown_config)!r}")

    graph = FaceGraph(centroids, adjacencies)
    starts = tuple(starts_value)
    graph.validate_terminals(goal, starts)
    config = SolverConfig(**config_value)
    return graph, goal, starts, config


def solution_to_dict(graph: FaceGraph, solution: TreeSolution) -> dict[str, Any]:
    return {
        "goal": solution.goal,
        "starts": list(solution.starts),
        "cost": solution.cost,
        "nodes": sorted(solution.nodes),
        "edges": [
            {
                "parent": parent,
                "child": child,
                "weight": graph.weight(parent, child),
            }
            for parent, child in solution.oriented_edges()
        ],
    }
