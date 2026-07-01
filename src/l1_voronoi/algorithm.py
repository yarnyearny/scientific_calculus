"""Algorithms for obstacle-aware L1 Voronoi subdivision on a rectilinear grid.

The implementation follows four explicit phases:

1. Define a problem with sites, an axis-aligned rectangular domain, and
   axis-aligned segment obstacles.
2. Propagate shortest L1 distances from all sites over the Hanan grid induced by
   the domain, sites, and obstacle endpoints.
3. Divide each obstacle-free unit cell using only the weighted distances from
   its four corners.
4. Return all local cell pieces in global coordinates so adjacent cells can be
   consumed together by callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf, isclose
from typing import Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

EPSILON = 1e-9


@dataclass(frozen=True, order=True)
class Point:
    """A point in the plane."""

    x: float
    y: float


@dataclass(frozen=True)
class Segment:
    """An axis-aligned obstacle segment."""

    a: Point
    b: Point

    def __post_init__(self) -> None:
        if not (isclose(self.a.x, self.b.x, abs_tol=EPSILON) or isclose(self.a.y, self.b.y, abs_tol=EPSILON)):
            raise ValueError("obstacle segments must be axis-aligned")
        if self.a == self.b:
            raise ValueError("obstacle segments must have positive length")

    @property
    def is_vertical(self) -> bool:
        return isclose(self.a.x, self.b.x, abs_tol=EPSILON)

    @property
    def is_horizontal(self) -> bool:
        return isclose(self.a.y, self.b.y, abs_tol=EPSILON)

    @property
    def min_x(self) -> float:
        return min(self.a.x, self.b.x)

    @property
    def max_x(self) -> float:
        return max(self.a.x, self.b.x)

    @property
    def min_y(self) -> float:
        return min(self.a.y, self.b.y)

    @property
    def max_y(self) -> float:
        return max(self.a.y, self.b.y)

    def contains_interior_point(self, p: Point) -> bool:
        """Return whether p lies in the relative interior of this segment."""

        if self.is_vertical:
            return (
                isclose(p.x, self.a.x, abs_tol=EPSILON)
                and self.min_y + EPSILON < p.y < self.max_y - EPSILON
            )
        return (
            isclose(p.y, self.a.y, abs_tol=EPSILON)
            and self.min_x + EPSILON < p.x < self.max_x - EPSILON
        )


@dataclass(frozen=True)
class L1VoronoiProblem:
    """Input for the grid-based Voronoi algorithm.

    Attributes:
        domain: ``(min_x, min_y, max_x, max_y)`` rectangular bounding box.
        sites: Mapping from stable site identifiers to site points.
        obstacles: Axis-aligned segment obstacles that cannot be crossed.
    """

    domain: Tuple[float, float, float, float]
    sites: Mapping[str, Point]
    obstacles: Tuple[Segment, ...] = ()


@dataclass(frozen=True)
class VertexLabel:
    """Shortest propagated distance and all sites that attain it."""

    distance: float
    sites: FrozenSet[str]


@dataclass(frozen=True)
class CellRegion:
    """A polygonal piece of a unit cell owned by the sites attached to a corner."""

    corner: Point
    distance: float
    sites: FrozenSet[str]
    polygon: Tuple[Point, ...]


@dataclass(frozen=True)
class CellDivision:
    """Voronoi subdivision of one grid cell."""

    lower_left: Point
    upper_right: Point
    regions: Tuple[CellRegion, ...]

    def nearest_sites_at(self, p: Point) -> FrozenSet[str]:
        """Return the nearest site set at p according to this cell's corner data."""

        if not (
            self.lower_left.x - EPSILON <= p.x <= self.upper_right.x + EPSILON
            and self.lower_left.y - EPSILON <= p.y <= self.upper_right.y + EPSILON
        ):
            raise ValueError("point is outside this cell")

        best = inf
        winners: set[str] = set()
        for region in self.regions:
            value = region.distance + l1_distance(region.corner, p)
            if value < best - EPSILON:
                best = value
                winners = set(region.sites)
            elif isclose(value, best, abs_tol=EPSILON):
                winners.update(region.sites)
        return frozenset(winners)


@dataclass(frozen=True)
class Diagram:
    """Complete result of the algorithm."""

    x_coordinates: Tuple[float, ...]
    y_coordinates: Tuple[float, ...]
    vertex_labels: Mapping[Point, VertexLabel]
    cells: Tuple[CellDivision, ...]

    def nearest_sites_at(self, p: Point) -> FrozenSet[str]:
        """Return nearest sites at p by locating its grid cell."""

        for cell in self.cells:
            if (
                cell.lower_left.x - EPSILON <= p.x <= cell.upper_right.x + EPSILON
                and cell.lower_left.y - EPSILON <= p.y <= cell.upper_right.y + EPSILON
            ):
                return cell.nearest_sites_at(p)
        raise ValueError("point is outside all free cells")


def compute_diagram(problem: L1VoronoiProblem) -> Diagram:
    """Compute an obstacle-aware L1 Voronoi diagram.

    The grid includes the domain bounds, every site coordinate, and every
    obstacle endpoint coordinate. Sites therefore become grid vertices, and each
    cell can be divided from its four corner labels.
    """

    _validate_problem(problem)
    x_coordinates, y_coordinates = _grid_coordinates(problem)
    blocked_vertices = _blocked_vertices(x_coordinates, y_coordinates, problem.obstacles)
    vertex_labels = _propagate(problem, x_coordinates, y_coordinates, blocked_vertices)
    cells = tuple(_divide_cells(x_coordinates, y_coordinates, vertex_labels, problem.obstacles))
    return Diagram(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        vertex_labels=vertex_labels,
        cells=cells,
    )


def l1_distance(a: Point, b: Point) -> float:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _validate_problem(problem: L1VoronoiProblem) -> None:
    min_x, min_y, max_x, max_y = problem.domain
    if not (min_x < max_x and min_y < max_y):
        raise ValueError("domain must be (min_x, min_y, max_x, max_y)")
    if not problem.sites:
        raise ValueError("at least one site is required")
    for site_id, site in problem.sites.items():
        if not site_id:
            raise ValueError("site identifiers must be non-empty")
        if not (min_x - EPSILON <= site.x <= max_x + EPSILON and min_y - EPSILON <= site.y <= max_y + EPSILON):
            raise ValueError(f"site {site_id!r} lies outside the domain")
    for obstacle in problem.obstacles:
        for endpoint in (obstacle.a, obstacle.b):
            if not (
                min_x - EPSILON <= endpoint.x <= max_x + EPSILON
                and min_y - EPSILON <= endpoint.y <= max_y + EPSILON
            ):
                raise ValueError("obstacle endpoints must lie inside the domain")


def _grid_coordinates(problem: L1VoronoiProblem) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    min_x, min_y, max_x, max_y = problem.domain
    xs = {min_x, max_x}
    ys = {min_y, max_y}

    for site in problem.sites.values():
        xs.add(site.x)
        ys.add(site.y)
    for obstacle in problem.obstacles:
        xs.add(obstacle.a.x)
        xs.add(obstacle.b.x)
        ys.add(obstacle.a.y)
        ys.add(obstacle.b.y)

    return tuple(sorted(xs)), tuple(sorted(ys))


def _blocked_vertices(
    x_coordinates: Sequence[float],
    y_coordinates: Sequence[float],
    obstacles: Sequence[Segment],
) -> FrozenSet[Point]:
    blocked: set[Point] = set()
    for x in x_coordinates:
        for y in y_coordinates:
            p = Point(x, y)
            if any(obstacle.contains_interior_point(p) for obstacle in obstacles):
                blocked.add(p)
    return frozenset(blocked)


def _propagate(
    problem: L1VoronoiProblem,
    x_coordinates: Sequence[float],
    y_coordinates: Sequence[float],
    blocked_vertices: FrozenSet[Point],
) -> Dict[Point, VertexLabel]:
    site_at_point: Dict[Point, set[str]] = {}
    for site_id, site in problem.sites.items():
        if site in blocked_vertices:
            raise ValueError(f"site {site_id!r} lies in the interior of an obstacle")
        site_at_point.setdefault(site, set()).add(site_id)

    points = [Point(x, y) for x in x_coordinates for y in y_coordinates if Point(x, y) not in blocked_vertices]
    distances: Dict[Point, float] = {point: inf for point in points}
    labels: Dict[Point, set[str]] = {point: set() for point in points}
    queue: List[Tuple[float, int, Point]] = []
    sequence = 0

    for point, sites in site_at_point.items():
        distances[point] = 0.0
        labels[point].update(sites)
        heappush(queue, (0.0, sequence, point))
        sequence += 1

    while queue:
        distance, _, point = heappop(queue)
        if distance > distances[point] + EPSILON:
            continue

        for neighbor in _neighbors(point, x_coordinates, y_coordinates, blocked_vertices, problem.obstacles):
            next_distance = distance + l1_distance(point, neighbor)
            if next_distance < distances[neighbor] - EPSILON:
                distances[neighbor] = next_distance
                labels[neighbor] = set(labels[point])
                heappush(queue, (next_distance, sequence, neighbor))
                sequence += 1
            elif isclose(next_distance, distances[neighbor], abs_tol=EPSILON):
                before = len(labels[neighbor])
                labels[neighbor].update(labels[point])
                if len(labels[neighbor]) != before:
                    heappush(queue, (next_distance, sequence, neighbor))
                    sequence += 1

    return {
        point: VertexLabel(distance=distance, sites=frozenset(labels[point]))
        for point, distance in distances.items()
        if distance < inf and labels[point]
    }


def _neighbors(
    point: Point,
    x_coordinates: Sequence[float],
    y_coordinates: Sequence[float],
    blocked_vertices: FrozenSet[Point],
    obstacles: Sequence[Segment],
) -> Iterator[Point]:
    xi = x_coordinates.index(point.x)
    yi = y_coordinates.index(point.y)
    candidate_indices = ((xi - 1, yi), (xi + 1, yi), (xi, yi - 1), (xi, yi + 1))

    for nx, ny in candidate_indices:
        if nx < 0 or nx >= len(x_coordinates) or ny < 0 or ny >= len(y_coordinates):
            continue
        neighbor = Point(x_coordinates[nx], y_coordinates[ny])
        if neighbor in blocked_vertices:
            continue
        if not _edge_blocked(point, neighbor, obstacles):
            yield neighbor


def _edge_blocked(a: Point, b: Point, obstacles: Sequence[Segment]) -> bool:
    for obstacle in obstacles:
        if _open_segments_intersect(a, b, obstacle):
            return True
    return False


def _open_segments_intersect(a: Point, b: Point, obstacle: Segment) -> bool:
    """Return whether open segment ab intersects the obstacle interior.

    Endpoint contacts are allowed so shortest paths can bend around obstacle
    endpoints. Moving along the obstacle interior is blocked.
    """

    horizontal_edge = isclose(a.y, b.y, abs_tol=EPSILON)
    vertical_edge = isclose(a.x, b.x, abs_tol=EPSILON)

    if horizontal_edge and obstacle.is_vertical:
        y = a.y
        x1, x2 = sorted((a.x, b.x))
        x = obstacle.a.x
        return (
            x1 + EPSILON < x < x2 - EPSILON
            and obstacle.min_y + EPSILON < y < obstacle.max_y - EPSILON
        )

    if vertical_edge and obstacle.is_horizontal:
        x = a.x
        y1, y2 = sorted((a.y, b.y))
        y = obstacle.a.y
        return (
            y1 + EPSILON < y < y2 - EPSILON
            and obstacle.min_x + EPSILON < x < obstacle.max_x - EPSILON
        )

    if vertical_edge and obstacle.is_vertical and isclose(a.x, obstacle.a.x, abs_tol=EPSILON):
        y1, y2 = sorted((a.y, b.y))
        return max(y1, obstacle.min_y) < min(y2, obstacle.max_y) - EPSILON

    if horizontal_edge and obstacle.is_horizontal and isclose(a.y, obstacle.a.y, abs_tol=EPSILON):
        x1, x2 = sorted((a.x, b.x))
        return max(x1, obstacle.min_x) < min(x2, obstacle.max_x) - EPSILON

    return False


def _divide_cells(
    x_coordinates: Sequence[float],
    y_coordinates: Sequence[float],
    vertex_labels: Mapping[Point, VertexLabel],
    obstacles: Sequence[Segment],
) -> Iterator[CellDivision]:
    for ix in range(len(x_coordinates) - 1):
        for iy in range(len(y_coordinates) - 1):
            lower_left = Point(x_coordinates[ix], y_coordinates[iy])
            upper_right = Point(x_coordinates[ix + 1], y_coordinates[iy + 1])
            if _cell_blocked(lower_left, upper_right, obstacles):
                continue

            corners = (
                lower_left,
                Point(upper_right.x, lower_left.y),
                Point(lower_left.x, upper_right.y),
                upper_right,
            )
            candidates = [
                (corner, label)
                for corner in corners
                if (label := vertex_labels.get(corner)) is not None
            ]
            if not candidates:
                continue

            regions = []
            for corner, label in candidates:
                polygon = _corner_region_polygon(
                    lower_left,
                    upper_right,
                    corner,
                    label.distance,
                    [(other_corner, other_label.distance) for other_corner, other_label in candidates],
                )
                if _polygon_area(polygon) > EPSILON:
                    regions.append(
                        CellRegion(
                            corner=corner,
                            distance=label.distance,
                            sites=label.sites,
                            polygon=tuple(polygon),
                        )
                    )

            if regions:
                yield CellDivision(lower_left=lower_left, upper_right=upper_right, regions=tuple(regions))


def _cell_blocked(lower_left: Point, upper_right: Point, obstacles: Sequence[Segment]) -> bool:
    """Return whether a cell interior is occupied by an obstacle.

    With the induced grid, valid obstacles lie on cell boundaries, so a blocked
    cell indicates invalid input such as an obstacle through a cell interior.
    """

    for obstacle in obstacles:
        if obstacle.is_vertical and lower_left.x + EPSILON < obstacle.a.x < upper_right.x - EPSILON:
            if max(lower_left.y, obstacle.min_y) < min(upper_right.y, obstacle.max_y) - EPSILON:
                return True
        if obstacle.is_horizontal and lower_left.y + EPSILON < obstacle.a.y < upper_right.y - EPSILON:
            if max(lower_left.x, obstacle.min_x) < min(upper_right.x, obstacle.max_x) - EPSILON:
                return True
    return False


def _corner_region_polygon(
    lower_left: Point,
    upper_right: Point,
    corner: Point,
    corner_distance: float,
    candidates: Sequence[Tuple[Point, float]],
) -> List[Point]:
    polygon = [
        lower_left,
        Point(upper_right.x, lower_left.y),
        upper_right,
        Point(lower_left.x, upper_right.y),
    ]
    for other_corner, other_distance in candidates:
        if other_corner == corner:
            continue
        inequality = _candidate_inequality(
            lower_left,
            upper_right,
            corner,
            corner_distance,
            other_corner,
            other_distance,
        )
        polygon = _clip_polygon(polygon, inequality)
        if not polygon:
            return []
    return polygon


def _candidate_inequality(
    lower_left: Point,
    upper_right: Point,
    corner: Point,
    corner_distance: float,
    other_corner: Point,
    other_distance: float,
) -> Tuple[float, float, float]:
    """Return a*x + b*y + c <= 0 for candidate(corner) <= candidate(other)."""

    a1, b1, c1 = _linear_l1_from_corner(lower_left, upper_right, corner, corner_distance)
    a2, b2, c2 = _linear_l1_from_corner(lower_left, upper_right, other_corner, other_distance)
    return a1 - a2, b1 - b2, c1 - c2


def _linear_l1_from_corner(
    lower_left: Point,
    upper_right: Point,
    corner: Point,
    distance: float,
) -> Tuple[float, float, float]:
    """Return a*x + b*y + c for distance + |p - corner|_1 inside a cell."""

    if isclose(corner.x, lower_left.x, abs_tol=EPSILON):
        x_sign = 1.0
        x_constant = -corner.x
    elif isclose(corner.x, upper_right.x, abs_tol=EPSILON):
        x_sign = -1.0
        x_constant = corner.x
    else:
        raise ValueError("corner x-coordinate is not on this cell")

    if isclose(corner.y, lower_left.y, abs_tol=EPSILON):
        y_sign = 1.0
        y_constant = -corner.y
    elif isclose(corner.y, upper_right.y, abs_tol=EPSILON):
        y_sign = -1.0
        y_constant = corner.y
    else:
        raise ValueError("corner y-coordinate is not on this cell")

    return x_sign, y_sign, distance + x_constant + y_constant


def _clip_polygon(polygon: Sequence[Point], inequality: Tuple[float, float, float]) -> List[Point]:
    if not polygon:
        return []

    clipped: List[Point] = []
    previous = polygon[-1]
    previous_value = _evaluate(inequality, previous)
    previous_inside = previous_value <= EPSILON

    for current in polygon:
        current_value = _evaluate(inequality, current)
        current_inside = current_value <= EPSILON

        if current_inside != previous_inside:
            clipped.append(_intersection(previous, current, previous_value, current_value))
        if current_inside:
            clipped.append(current)

        previous = current
        previous_value = current_value
        previous_inside = current_inside

    return clipped


def _evaluate(inequality: Tuple[float, float, float], p: Point) -> float:
    a, b, c = inequality
    return a * p.x + b * p.y + c


def _intersection(a: Point, b: Point, value_a: float, value_b: float) -> Point:
    denominator = value_a - value_b
    if isclose(denominator, 0.0, abs_tol=EPSILON):
        return a
    t = value_a / denominator
    return Point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))


def _polygon_area(polygon: Sequence[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for current, next_point in zip(polygon, polygon[1:] + polygon[:1]):
        total += current.x * next_point.y - next_point.x * current.y
    return abs(total) / 2.0
