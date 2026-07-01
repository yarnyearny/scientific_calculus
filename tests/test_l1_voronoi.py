import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from l1_voronoi import L1VoronoiProblem, Point, Segment, compute_diagram


class L1VoronoiTests(unittest.TestCase):
    def test_unobstructed_cell_uses_corner_labels(self) -> None:
        diagram = compute_diagram(
            L1VoronoiProblem(
                domain=(0, 0, 4, 2),
                sites={
                    "left": Point(0, 0),
                    "right": Point(4, 0),
                },
            )
        )

        self.assertEqual(diagram.nearest_sites_at(Point(0.5, 1)), frozenset({"left"}))
        self.assertEqual(diagram.nearest_sites_at(Point(3.5, 1)), frozenset({"right"}))
        self.assertEqual(diagram.nearest_sites_at(Point(2, 1)), frozenset({"left", "right"}))

    def test_tied_vertex_keeps_all_sites(self) -> None:
        diagram = compute_diagram(
            L1VoronoiProblem(
                domain=(0, 0, 2, 2),
                sites={
                    "a": Point(0, 0),
                    "b": Point(2, 0),
                },
            )
        )

        label = diagram.vertex_labels[Point(1, 2)]
        self.assertEqual(label.distance, 3)
        self.assertEqual(label.sites, frozenset({"a", "b"}))

    def test_vertical_obstacle_forces_propagation_around_endpoint(self) -> None:
        diagram = compute_diagram(
            L1VoronoiProblem(
                domain=(0, 0, 2, 2),
                sites={
                    "left": Point(0, 1),
                    "right": Point(2, 1),
                },
                obstacles=(Segment(Point(1, 0), Point(1, 1.5)),),
            )
        )

        left_of_barrier = diagram.vertex_labels[Point(0, 2)]
        right_of_barrier = diagram.vertex_labels[Point(2, 2)]
        self.assertEqual(left_of_barrier.sites, frozenset({"left"}))
        self.assertEqual(right_of_barrier.sites, frozenset({"right"}))
        self.assertEqual(diagram.nearest_sites_at(Point(1.5, 0.5)), frozenset({"right"}))

    def test_cell_division_returns_polygons(self) -> None:
        diagram = compute_diagram(
            L1VoronoiProblem(
                domain=(0, 0, 2, 2),
                sites={
                    "lower_left": Point(0, 0),
                    "upper_right": Point(2, 2),
                },
            )
        )

        regions = [region for cell in diagram.cells for region in cell.regions]
        self.assertTrue(regions)
        self.assertTrue(all(len(region.polygon) >= 3 for region in regions))


if __name__ == "__main__":
    unittest.main()
