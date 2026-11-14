import unittest

from scientific_calculus.model import FaceGraph, TreeSolution


class FaceGraphTest(unittest.TestCase):
    def test_l1_weight_and_deduplicated_adjacency(self) -> None:
        graph = FaceGraph(
            {"a": (0, 0), "b": (2, -3)},
            [("a", "b"), ("b", "a")],
        )

        self.assertEqual(graph.weight("a", "b"), 5.0)
        self.assertEqual(graph.neighbors("a"), ("b",))
        self.assertEqual(graph.edges, frozenset({("a", "b")}))

    def test_rejects_unreachable_start(self) -> None:
        graph = FaceGraph(
            {"goal": (0, 0), "start": (1, 0), "isolated": (2, 0)},
            [("goal", "start")],
        )

        with self.assertRaisesRegex(ValueError, "到達できない"):
            graph.validate_terminals("goal", ("isolated",))


class TreeSolutionTest(unittest.TestCase):
    def test_orients_tree_from_goal_and_counts_shared_edge_once(self) -> None:
        graph = FaceGraph(
            {"g": (0, 0), "j": (1, 0), "a": (2, 1), "b": (2, -1)},
            [("g", "j"), ("j", "a"), ("j", "b")],
        )

        solution = TreeSolution.create(
            graph, "g", ("a", "b"), [("j", "g"), ("a", "j"), ("b", "j")]
        )

        self.assertEqual(solution.cost, 5.0)
        self.assertEqual(
            solution.oriented_edges(), (("g", "j"), ("j", "a"), ("j", "b"))
        )

    def test_rejects_cycle(self) -> None:
        graph = FaceGraph(
            {"a": (0, 0), "b": (1, 0), "c": (0, 1)},
            [("a", "b"), ("b", "c"), ("a", "c")],
        )

        with self.assertRaisesRegex(ValueError, "閉路"):
            TreeSolution.create(
                graph, "a", ("b", "c"), [("a", "b"), ("b", "c"), ("a", "c")]
            )


if __name__ == "__main__":
    unittest.main()
