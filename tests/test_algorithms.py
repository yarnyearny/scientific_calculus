import unittest

from scientific_calculus.construction import (
    metric_closure_tree,
    sequential_attachment,
)
from scientific_calculus.model import FaceGraph
from scientific_calculus.paths import (
    dijkstra,
    normalize_tree,
    shortest_path_to_any,
)
from scientific_calculus.solver import SolverConfig, solve


def shared_trunk_graph() -> FaceGraph:
    return FaceGraph(
        {
            "g": (0, 0),
            "j": (1, 0),
            "a": (2, 1),
            "b": (2, -1),
            "unused": (1, 2),
        },
        [
            ("g", "j"),
            ("j", "a"),
            ("j", "b"),
            ("g", "unused"),
            ("unused", "a"),
        ],
    )


class PathAlgorithmsTest(unittest.TestCase):
    def test_dijkstra_and_multi_target_path_are_deterministic(self) -> None:
        graph = shared_trunk_graph()

        distances, _ = dijkstra(graph, "g")

        self.assertEqual(distances["a"], 3.0)
        self.assertEqual(shortest_path_to_any(graph, "b", {"g", "a"}), ("b", "j", "g"))

    def test_normalize_removes_cycle_and_nonterminal_leaf(self) -> None:
        graph = FaceGraph(
            {
                "g": (0, 0),
                "x": (1, 0),
                "a": (2, 0),
                "leaf": (1, 1),
            },
            [("g", "x"), ("x", "a"), ("g", "a"), ("x", "leaf")],
        )

        result = normalize_tree(
            graph,
            [("g", "x"), ("x", "a"), ("g", "a"), ("x", "leaf")],
            {"g", "a"},
        )

        self.assertNotIn(("leaf", "x"), result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result, frozenset({("a", "x"), ("g", "x")}))


class ConstructionTest(unittest.TestCase):
    def test_sequential_attachment_builds_shared_trunk(self) -> None:
        graph = shared_trunk_graph()

        solution = sequential_attachment(
            graph, "g", ("a", "b"), ("a", "b")
        )

        self.assertEqual(
            solution.edges,
            frozenset({("g", "j"), ("a", "j"), ("b", "j")}),
        )
        self.assertEqual(solution.cost, 5.0)

    def test_metric_closure_returns_a_valid_tree(self) -> None:
        graph = shared_trunk_graph()

        solution = metric_closure_tree(graph, "g", ("a", "b"))

        self.assertEqual(len(solution.edges), len(solution.nodes) - 1)
        self.assertTrue({"g", "a", "b"} <= solution.nodes)
        self.assertEqual(solution.cost, 5.0)

    def test_solver_is_reproducible_and_returns_consistent_cost(self) -> None:
        graph = shared_trunk_graph()
        config = SolverConfig(random_restarts=8, local_iterations=30, seed=42)

        first = solve(graph, "g", ("a", "b"), config)
        second = solve(graph, "g", ("a", "b"), config)

        self.assertEqual(first, second)
        self.assertEqual(
            first.cost, sum(graph.weight(*edge) for edge in first.edges)
        )
        self.assertEqual(len(first.edges), len(first.nodes) - 1)


if __name__ == "__main__":
    unittest.main()
