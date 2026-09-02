"""Independent Cartesian surface-charge BEM for QD--spheroid validation.

This module is deliberately independent of :mod:`qd_mnp_spheroid_green` and
of all spheroidal-harmonic formulae.  It discretizes the dielectric boundary
integral equation directly on a triangulated surface,

    [Lambda I + K*] sigma = n . E_inc,

where ``Lambda=(eps_p+eps_m)/(2*(eps_p-eps_m))`` and

    K* sigma(x) = PV integral n(x).(y-x) sigma(y)
                  / (4*pi*|x-y|**3) dS_y.

The implementation is a validation baseline, not a production high-order
BEM.  It uses constant density and centroid collocation on flat triangles.
The principal-value contribution of a triangle to its own centroid is zero;
the analytic jump is the ``Lambda I`` term.  Near-singular observation
integrals consequently converge only with mesh refinement.  Quantitative
use, especially for a small QD--surface gap, must use the nested-mesh audit
provided below or an offline higher-order/adaptive quadrature implementation.

Units follow the atomic-unit electrostatic convention of the project: the
usual ``4*pi*epsilon_0`` factor is absent, while a dipole in a host produces
``Phi=p.r/(eps_m*r**3)``.  No project physics module is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgWarning, solve


FOUR_PI = 4.0 * np.pi
MAX_DENSE_PANELS = 6_000


def _readonly(value: ArrayLike, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_positive_scalar(name: str, value: float) -> float:
    if not np.isscalar(value) or not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return float(value)


def _real_vector(name: str, value: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.shape != (3,) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real vector with shape (3,).")
    result = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


@dataclass(frozen=True)
class TriangularSurfaceMesh:
    """Flat triangular approximation to one axis-aligned spheroid."""

    a_au: float
    c_au: float
    subdivision_level: int
    vertices_au: np.ndarray
    faces: np.ndarray
    centroids_au: np.ndarray
    outward_normals: np.ndarray
    areas_au2: np.ndarray
    max_edge_au: float

    def __post_init__(self) -> None:
        a = _finite_positive_scalar("a_au", self.a_au)
        c = _finite_positive_scalar("c_au", self.c_au)
        if c < a:
            raise ValueError("The validation mesh requires a prolate or spherical body, c_au >= a_au.")
        if (
            isinstance(self.subdivision_level, (bool, np.bool_))
            or not isinstance(self.subdivision_level, (int, np.integer))
            or self.subdivision_level < 0
        ):
            raise ValueError("subdivision_level must be an integer >= 0.")

        vertices = _readonly(self.vertices_au, dtype=float)
        faces = _readonly(self.faces, dtype=int)
        centroids = _readonly(self.centroids_au, dtype=float)
        normals = _readonly(self.outward_normals, dtype=float)
        areas = _readonly(self.areas_au2, dtype=float)
        panel_count = faces.shape[0] if faces.ndim == 2 else -1
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices_au must have shape (vertex_count, 3).")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (panel_count, 3).")
        if centroids.shape != (panel_count, 3) or normals.shape != (panel_count, 3):
            raise ValueError("centroids and normals must have one 3-vector per panel.")
        if areas.shape != (panel_count,) or np.any(areas <= 0.0):
            raise ValueError("areas_au2 must contain one positive value per panel.")
        if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
            raise ValueError("faces contains an invalid vertex index.")
        arrays = (vertices, centroids, normals, areas)
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("Mesh geometry must contain only finite values.")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=2e-14, atol=2e-14):
            raise ValueError("outward_normals must be unit vectors.")
        max_edge = _finite_positive_scalar("max_edge_au", self.max_edge_au)

        object.__setattr__(self, "a_au", a)
        object.__setattr__(self, "c_au", c)
        object.__setattr__(self, "subdivision_level", int(self.subdivision_level))
        object.__setattr__(self, "vertices_au", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "centroids_au", centroids)
        object.__setattr__(self, "outward_normals", normals)
        object.__setattr__(self, "areas_au2", areas)
        object.__setattr__(self, "max_edge_au", max_edge)

    @property
    def panel_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_au.shape[0])

    @property
    def surface_area_au2(self) -> float:
        return float(np.sum(self.areas_au2))


@dataclass(frozen=True)
class BEMObservables:
    """The A/B/K ports extracted without spheroidal modal formulae."""

    A_au3: complex
    B_field: complex
    B_dipole: complex
    K_au_minus3: complex

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.A_au3, self.B_field, self.B_dipole, self.K_au_minus3],
            dtype=complex,
        )
        if np.any(~np.isfinite(values)):
            raise FloatingPointError("BEM observables must be finite.")
        object.__setattr__(self, "A_au3", complex(self.A_au3))
        object.__setattr__(self, "B_field", complex(self.B_field))
        object.__setattr__(self, "B_dipole", complex(self.B_dipole))
        object.__setattr__(self, "K_au_minus3", complex(self.K_au_minus3))


@dataclass(frozen=True)
class BEMObservableErrors:
    """Non-negative scalar diagnostics corresponding to A/B_field/B_dipole/K."""

    A: float
    B_field: float
    B_dipole: float
    K: float

    def __post_init__(self) -> None:
        values = np.asarray([self.A, self.B_field, self.B_dipole, self.K], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("BEM observable errors must be finite and non-negative.")
        object.__setattr__(self, "A", float(self.A))
        object.__setattr__(self, "B_field", float(self.B_field))
        object.__setattr__(self, "B_dipole", float(self.B_dipole))
        object.__setattr__(self, "K", float(self.K))

    @property
    def maximum(self) -> float:
        return float(max(self.A, self.B_field, self.B_dipole, self.K))


@dataclass(frozen=True)
class BEMDiagnostics:
    """Discretization and algebra checks for one dense BEM solve."""

    panel_count: int
    vertex_count: int
    subdivision_level: int
    max_edge_au: float
    surface_area_au2: float
    minimum_qd_surface_distance_au: float
    qd_distance_over_max_edge: float
    bie_lambda: complex | None
    zero_contrast: bool
    relative_residual_uniform: float
    relative_residual_point_dipole: float
    net_charge_uniform: complex
    net_charge_point_dipole: complex
    relative_net_charge_uniform: float
    relative_net_charge_point_dipole: float
    reciprocity_absolute_error: float
    reciprocity_relative_error: float
    quadrature_rule: str = "constant_panel_centroid"


@dataclass(frozen=True)
class BEMResponse:
    mesh: TriangularSurfaceMesh
    observables: BEMObservables
    diagnostics: BEMDiagnostics


@dataclass(frozen=True)
class BEMConvergenceResult:
    """Nested-mesh values, extrapolation, and a numerical uncertainty audit.

    ``extrapolated`` is obtained from the finest ``extrapolation_levels`` by
    fitting the declared expansion

    ``X(h) = X(0) + c_1 h**p + c_2 h**(p+1) + ...``.

    The uncertainty is not fitted against an analytic spheroidal answer.  It
    is the larger of the fit residual and the change in ``X(0)`` when the
    highest correction term is removed.  This deliberately conservative
    model-stability estimate is useful for deciding whether a BEM fixture is
    independently resolved enough to validate an analytic kernel.
    """

    levels: tuple[int, ...]
    responses: tuple[BEMResponse, ...]
    assumed_order: float
    correction_orders: tuple[float, ...]
    extrapolation_levels: tuple[int, ...]
    extrapolation_h_scale_au: float
    extrapolated: BEMObservables
    lower_order_extrapolated: BEMObservables
    finest_relative_change: BEMObservableErrors
    extrapolation_relative_residual: BEMObservableErrors
    estimated_absolute_uncertainty: BEMObservableErrors
    estimated_relative_uncertainty: BEMObservableErrors
    uncertainty_method: str = "lower_order_stability_plus_fit_residual"

    def __post_init__(self) -> None:
        if len(self.levels) < 2 or len(self.levels) != len(self.responses):
            raise ValueError("A convergence result requires at least two matching levels and responses.")
        if tuple(sorted(set(self.levels))) != self.levels:
            raise ValueError("Convergence levels must be strictly increasing.")
        if not np.isfinite(self.assumed_order) or self.assumed_order <= 0.0:
            raise ValueError("assumed_order must be finite and positive.")
        if (
            not self.correction_orders
            or any(not np.isfinite(order) or order <= 0.0 for order in self.correction_orders)
            or tuple(sorted(set(self.correction_orders))) != self.correction_orders
        ):
            raise ValueError("correction_orders must be finite, positive, and increasing.")
        if len(self.extrapolation_levels) < len(self.correction_orders) + 1:
            raise ValueError(
                "extrapolation_levels must outnumber the fitted correction terms."
            )
        if tuple(self.levels[-len(self.extrapolation_levels) :]) != self.extrapolation_levels:
            raise ValueError("extrapolation_levels must be the finest convergence levels.")
        if not np.isfinite(self.extrapolation_h_scale_au) or self.extrapolation_h_scale_au <= 0.0:
            raise ValueError("extrapolation_h_scale_au must be finite and positive.")
        if not self.uncertainty_method:
            raise ValueError("uncertainty_method must be non-empty.")

    @property
    def maximum_finest_relative_change(self) -> float:
        return self.finest_relative_change.maximum

    @property
    def maximum_estimated_relative_uncertainty(self) -> float:
        return self.estimated_relative_uncertainty.maximum


def _base_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = np.asarray(
        [
            (-1.0, phi, 0.0),
            (1.0, phi, 0.0),
            (-1.0, -phi, 0.0),
            (1.0, -phi, 0.0),
            (0.0, -1.0, phi),
            (0.0, 1.0, phi),
            (0.0, -1.0, -phi),
            (0.0, 1.0, -phi),
            (phi, 0.0, -1.0),
            (phi, 0.0, 1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0, 1.0),
        ],
        dtype=float,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=int,
    )
    return vertices, faces


def _subdivide_unit_sphere(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertex_list = [vertex.copy() for vertex in vertices]
    midpoint_indices: dict[tuple[int, int], int] = {}

    def midpoint(left: int, right: int) -> int:
        edge = (left, right) if left < right else (right, left)
        cached = midpoint_indices.get(edge)
        if cached is not None:
            return cached
        value = vertices[left] + vertices[right]
        value /= np.linalg.norm(value)
        index = len(vertex_list)
        vertex_list.append(value)
        midpoint_indices[edge] = index
        return index

    refined: list[tuple[int, int, int]] = []
    for first, second, third in faces:
        ab = midpoint(int(first), int(second))
        bc = midpoint(int(second), int(third))
        ca = midpoint(int(third), int(first))
        refined.extend(
            [
                (int(first), ab, ca),
                (int(second), bc, ab),
                (int(third), ca, bc),
                (ab, bc, ca),
            ]
        )
    return np.asarray(vertex_list, dtype=float), np.asarray(refined, dtype=int)


def build_affine_icosphere(
    a_au: float,
    c_au: float,
    *,
    subdivision_level: int = 2,
) -> TriangularSurfaceMesh:
    """Create a deterministic flat triangular mesh of ``x²/a²+y²/a²+z²/c²=1``."""

    a = _finite_positive_scalar("a_au", a_au)
    c = _finite_positive_scalar("c_au", c_au)
    if c < a:
        raise ValueError("build_affine_icosphere requires c_au >= a_au.")
    if (
        isinstance(subdivision_level, (bool, np.bool_))
        or not isinstance(subdivision_level, (int, np.integer))
        or subdivision_level < 0
    ):
        raise ValueError("subdivision_level must be an integer >= 0.")

    unit_vertices, faces = _base_icosahedron()
    for _ in range(int(subdivision_level)):
        unit_vertices, faces = _subdivide_unit_sphere(unit_vertices, faces)

    vertices = unit_vertices * np.asarray([a, a, c], dtype=float)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centroids = np.mean(triangles, axis=1)
    ellipsoid_gradient = centroids / np.asarray([a * a, a * a, c * c])
    inward = np.einsum("ij,ij->i", cross, ellipsoid_gradient) < 0.0
    if np.any(inward):
        faces = faces.copy()
        faces[inward] = faces[inward][:, [0, 2, 1]]
        triangles = vertices[faces]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        centroids = np.mean(triangles, axis=1)

    double_area = np.linalg.norm(cross, axis=1)
    if np.any(double_area <= 0.0) or np.any(~np.isfinite(double_area)):
        raise FloatingPointError("The affine icosphere contains a degenerate triangle.")
    areas = 0.5 * double_area
    normals = cross / double_area[:, None]
    edge_lengths = np.concatenate(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ]
    )
    return TriangularSurfaceMesh(
        a_au=a,
        c_au=c,
        subdivision_level=int(subdivision_level),
        vertices_au=vertices,
        faces=faces,
        centroids_au=centroids,
        outward_normals=normals,
        areas_au2=areas,
        max_edge_au=float(np.max(edge_lengths)),
    )


def mesh_sha256(mesh: TriangularSurfaceMesh) -> str:
    """Return a portable identity hash for the deterministic surface mesh.

    Coordinates are rounded to 15 decimal places before hashing.  The mesh
    recipe is deterministic, but this canonicalization prevents harmless
    last-bit differences in platform ``sqrt`` implementations from changing
    fixture provenance.  Connectivity remains exact signed little-endian
    64-bit integer data.
    """

    if not isinstance(mesh, TriangularSurfaceMesh):
        raise TypeError("mesh must be a TriangularSurfaceMesh.")
    metadata = np.asarray(
        [mesh.a_au, mesh.c_au, float(mesh.subdivision_level)],
        dtype="<f8",
    )
    vertices = np.asarray(np.round(mesh.vertices_au, 15), dtype="<f8", order="C")
    faces = np.asarray(mesh.faces, dtype="<i8", order="C")
    digest = hashlib.sha256()
    digest.update(b"qdmnp-affine-icosphere-v1\0")
    digest.update(metadata.tobytes(order="C"))
    digest.update(np.asarray(vertices.shape, dtype="<i8").tobytes(order="C"))
    digest.update(vertices.tobytes(order="C"))
    digest.update(np.asarray(faces.shape, dtype="<i8").tobytes(order="C"))
    digest.update(faces.tobytes(order="C"))
    return digest.hexdigest()


def _validate_block_size(block_size: int) -> int:
    if (
        isinstance(block_size, (bool, np.bool_))
        or not isinstance(block_size, (int, np.integer))
        or block_size < 1
    ):
        raise ValueError("block_size must be an integer >= 1.")
    return int(block_size)


def _kernel_blocks(mesh: TriangularSurfaceMesh, block_size: int):
    """Yield row slices and centroid Nyström blocks of K*."""

    centers = mesh.centroids_au
    normals = mesh.outward_normals
    areas = mesh.areas_au2
    panel_count = mesh.panel_count
    for start in range(0, panel_count, block_size):
        stop = min(panel_count, start + block_size)
        displacement = centers[None, :, :] - centers[start:stop, None, :]
        radius_squared = np.einsum("bji,bji->bj", displacement, displacement)
        numerator = np.einsum(
            "bi,bji->bj",
            normals[start:stop],
            displacement,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            block = numerator * areas[None, :] / (
                FOUR_PI * radius_squared * np.sqrt(radius_squared)
            )
        local_rows = np.arange(stop - start)
        block[local_rows, np.arange(start, stop)] = 0.0
        yield slice(start, stop), block


def _assemble_bie_matrix(
    mesh: TriangularSurfaceMesh,
    bie_lambda: complex,
    *,
    block_size: int,
) -> np.ndarray:
    matrix = np.empty((mesh.panel_count, mesh.panel_count), dtype=complex, order="F")
    for row_slice, block in _kernel_blocks(mesh, block_size):
        matrix[row_slice, :] = block
    diagonal = np.diag_indices(mesh.panel_count)
    matrix[diagonal] += bie_lambda
    if np.any(~np.isfinite(matrix)):
        raise FloatingPointError("The dense BEM matrix contains non-finite values.")
    return matrix


def _apply_bie_operator(
    mesh: TriangularSurfaceMesh,
    densities: np.ndarray,
    bie_lambda: complex,
    *,
    block_size: int,
) -> np.ndarray:
    result = bie_lambda * densities
    for row_slice, block in _kernel_blocks(mesh, block_size):
        result[row_slice] += block @ densities
    return result


def _point_dipole_field(
    points_au: np.ndarray,
    source_position_au: np.ndarray,
    dipole: np.ndarray,
    eps_m: float,
) -> np.ndarray:
    displacement = points_au - source_position_au[None, :]
    radius_squared = np.einsum("ij,ij->i", displacement, displacement)
    if np.any(radius_squared <= 0.0):
        raise ValueError("The point dipole cannot lie on a collocation point.")
    inverse_radius_cubed = radius_squared ** -1.5
    projection = displacement @ dipole
    return (
        3.0 * displacement * (projection / radius_squared)[:, None]
        - dipole[None, :]
    ) * (inverse_radius_cubed / eps_m)[:, None]


def _surface_dipole(
    mesh: TriangularSurfaceMesh,
    density: np.ndarray,
    eps_m: float,
) -> np.ndarray:
    return (
        eps_m
        / FOUR_PI
        * np.sum(
            mesh.centroids_au
            * (density * mesh.areas_au2)[:, None],
            axis=0,
        )
    )


def _surface_field_at_point(
    mesh: TriangularSurfaceMesh,
    density: np.ndarray,
    point_au: np.ndarray,
) -> np.ndarray:
    displacement = point_au[None, :] - mesh.centroids_au
    radius_squared = np.einsum("ij,ij->i", displacement, displacement)
    if np.any(radius_squared <= 0.0):
        raise ValueError("The observation point cannot equal a panel centroid.")
    weights = density * mesh.areas_au2 / (FOUR_PI * radius_squared ** 1.5)
    return np.sum(displacement * weights[:, None], axis=0)


def _point_triangle_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    """Distance from a point to a triangle, following the Voronoi-region test."""

    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(point - (a + v * ab)))

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(point - (a + w * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(point - (b + w * (c - b))))

    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    closest = a + ab * v + ac * w
    return float(np.linalg.norm(point - closest))


def _minimum_surface_distance(mesh: TriangularSurfaceMesh, point: np.ndarray) -> float:
    triangles = mesh.vertices_au[mesh.faces]
    return min(_point_triangle_distance(point, triangle) for triangle in triangles)


def _relative_residual(applied: np.ndarray, rhs: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    return float(np.linalg.norm(applied - rhs) / denominator)


def _relative_net_charge(
    density: np.ndarray,
    areas: np.ndarray,
) -> tuple[complex, float]:
    net = complex(np.dot(areas, density))
    absolute_mass = float(np.dot(areas, np.abs(density)))
    relative = abs(net) / max(absolute_mass, np.finfo(float).tiny)
    return net, float(relative)


def solve_spheroid_bem_on_mesh(
    mesh: TriangularSurfaceMesh,
    *,
    qd_position_au: ArrayLike,
    polarization: ArrayLike,
    eps_m: float,
    epsilon_particle: complex,
    block_size: int = 128,
    max_dense_panels: int = MAX_DENSE_PANELS,
) -> BEMResponse:
    """Solve uniform-field and point-dipole excitations on one fixed mesh.

    ``polarization`` is normalized internally.  The uniform incident field and
    the unit point dipole use this same vector, but the QD position itself is a
    free three-dimensional point outside the spheroid.
    """

    if not isinstance(mesh, TriangularSurfaceMesh):
        raise TypeError("mesh must be a TriangularSurfaceMesh.")
    eps_host = _finite_positive_scalar("eps_m", eps_m)
    epsilon = np.asarray(epsilon_particle)
    if epsilon.ndim != 0:
        raise ValueError("epsilon_particle must be a finite scalar.")
    epsilon = complex(epsilon)
    if not np.isfinite(epsilon):
        raise ValueError("epsilon_particle must be finite.")
    qd_position = _real_vector("qd_position_au", qd_position_au)
    direction = _real_vector("polarization", polarization)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0:
        raise ValueError("polarization must be nonzero.")
    direction /= direction_norm
    block = _validate_block_size(block_size)
    if (
        isinstance(max_dense_panels, (bool, np.bool_))
        or not isinstance(max_dense_panels, (int, np.integer))
        or max_dense_panels < 1
    ):
        raise ValueError("max_dense_panels must be an integer >= 1.")
    if mesh.panel_count > max_dense_panels:
        estimated_gib = 16.0 * mesh.panel_count**2 / 2.0**30
        raise ValueError(
            f"Dense BEM mesh has {mesh.panel_count} panels and needs at least "
            f"{estimated_gib:.2f} GiB for one complex matrix; max_dense_panels="
            f"{max_dense_panels}. Use an offline matrix-free or compressed solver."
        )

    ellipsoid_level = np.sqrt(
        (qd_position[0] ** 2 + qd_position[1] ** 2) / mesh.a_au**2
        + qd_position[2] ** 2 / mesh.c_au**2
    )
    if ellipsoid_level <= 1.0:
        raise ValueError("qd_position_au must lie strictly outside the spheroid.")
    minimum_distance = _minimum_surface_distance(mesh, qd_position)
    if not np.isfinite(minimum_distance) or minimum_distance <= 0.0:
        raise ValueError("The QD point must lie strictly outside the triangulated surface.")

    rhs_uniform = mesh.outward_normals @ direction
    point_field = _point_dipole_field(
        mesh.centroids_au,
        qd_position,
        direction,
        eps_host,
    )
    rhs_dipole = np.einsum("ij,ij->i", mesh.outward_normals, point_field)
    rhs = np.column_stack((rhs_uniform, rhs_dipole)).astype(complex)
    contrast = epsilon - eps_host

    if contrast == 0.0:
        densities = np.zeros_like(rhs)
        bie_lambda: complex | None = None
        residual_uniform = 0.0
        residual_dipole = 0.0
    else:
        bie_lambda = (epsilon + eps_host) / (2.0 * contrast)
        matrix = _assemble_bie_matrix(mesh, bie_lambda, block_size=block)
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            try:
                densities = solve(
                    matrix,
                    rhs,
                    assume_a="gen",
                    overwrite_a=True,
                    overwrite_b=False,
                    check_finite=False,
                )
            except (LinAlgWarning, np.linalg.LinAlgError) as exc:
                raise np.linalg.LinAlgError(
                    "The dense dielectric BEM system is singular or numerically ill-conditioned."
                ) from exc
        if np.any(~np.isfinite(densities)):
            raise FloatingPointError("The dense BEM solve returned non-finite surface density.")
        applied = _apply_bie_operator(
            mesh,
            densities,
            bie_lambda,
            block_size=block,
        )
        residual_uniform = _relative_residual(applied[:, 0], rhs[:, 0])
        residual_dipole = _relative_residual(applied[:, 1], rhs[:, 1])

    density_uniform = densities[:, 0]
    density_dipole = densities[:, 1]
    particle_dipole_uniform = _surface_dipole(mesh, density_uniform, eps_host)
    particle_dipole_from_qd = _surface_dipole(mesh, density_dipole, eps_host)
    field_from_uniform = _surface_field_at_point(mesh, density_uniform, qd_position)
    field_from_qd = _surface_field_at_point(mesh, density_dipole, qd_position)

    observables = BEMObservables(
        A_au3=complex(np.dot(direction, particle_dipole_uniform)),
        B_field=complex(np.dot(direction, field_from_uniform)),
        B_dipole=complex(np.dot(direction, particle_dipole_from_qd)),
        K_au_minus3=complex(np.dot(direction, field_from_qd)),
    )
    net_uniform, relative_net_uniform = _relative_net_charge(
        density_uniform,
        mesh.areas_au2,
    )
    net_dipole, relative_net_dipole = _relative_net_charge(
        density_dipole,
        mesh.areas_au2,
    )
    reciprocity_absolute = abs(observables.B_field - observables.B_dipole)
    reciprocity_scale = max(
        abs(observables.B_field),
        abs(observables.B_dipole),
        np.finfo(float).tiny,
    )
    diagnostics = BEMDiagnostics(
        panel_count=mesh.panel_count,
        vertex_count=mesh.vertex_count,
        subdivision_level=mesh.subdivision_level,
        max_edge_au=mesh.max_edge_au,
        surface_area_au2=mesh.surface_area_au2,
        minimum_qd_surface_distance_au=float(minimum_distance),
        qd_distance_over_max_edge=float(minimum_distance / mesh.max_edge_au),
        bie_lambda=bie_lambda,
        zero_contrast=bool(contrast == 0.0),
        relative_residual_uniform=residual_uniform,
        relative_residual_point_dipole=residual_dipole,
        net_charge_uniform=net_uniform,
        net_charge_point_dipole=net_dipole,
        relative_net_charge_uniform=relative_net_uniform,
        relative_net_charge_point_dipole=relative_net_dipole,
        reciprocity_absolute_error=float(reciprocity_absolute),
        reciprocity_relative_error=float(reciprocity_absolute / reciprocity_scale),
    )
    return BEMResponse(mesh=mesh, observables=observables, diagnostics=diagnostics)


def solve_spheroid_bem(
    *,
    a_au: float,
    c_au: float,
    qd_position_au: ArrayLike,
    polarization: ArrayLike,
    eps_m: float,
    epsilon_particle: complex,
    subdivision_level: int = 2,
    block_size: int = 128,
    max_dense_panels: int = MAX_DENSE_PANELS,
) -> BEMResponse:
    """Build one affine icosphere mesh and evaluate the independent A/B/K ports."""

    mesh = build_affine_icosphere(
        a_au,
        c_au,
        subdivision_level=subdivision_level,
    )
    return solve_spheroid_bem_on_mesh(
        mesh,
        qd_position_au=qd_position_au,
        polarization=polarization,
        eps_m=eps_m,
        epsilon_particle=epsilon_particle,
        block_size=block_size,
        max_dense_panels=max_dense_panels,
    )


def _observable_array(observables: BEMObservables) -> np.ndarray:
    return np.asarray(
        [
            observables.A_au3,
            observables.B_field,
            observables.B_dipole,
            observables.K_au_minus3,
        ],
        dtype=complex,
    )


def _observables_from_array(values: ArrayLike) -> BEMObservables:
    array = np.asarray(values, dtype=complex)
    if array.shape != (4,):
        raise ValueError("An observable array must have shape (4,).")
    return BEMObservables(
        A_au3=array[0],
        B_field=array[1],
        B_dipole=array[2],
        K_au_minus3=array[3],
    )


def _errors_from_array(values: np.ndarray) -> BEMObservableErrors:
    return BEMObservableErrors(
        A=float(values[0]),
        B_field=float(values[1]),
        B_dipole=float(values[2]),
        K=float(values[3]),
    )


def run_nested_bem_convergence(
    *,
    a_au: float,
    c_au: float,
    qd_position_au: ArrayLike,
    polarization: ArrayLike,
    eps_m: float,
    epsilon_particle: complex,
    subdivision_levels: tuple[int, ...] = (1, 2, 3),
    assumed_order: float = 1.0,
    correction_terms: int = 2,
    extrapolation_level_count: int | None = None,
    block_size: int = 128,
    max_dense_panels: int = MAX_DENSE_PANELS,
) -> BEMConvergenceResult:
    """Run nested meshes and extrapolate a declared asymptotic h expansion.

    The default model is ``X(h)=X(0)+c1*h+c2*h**2`` on the three finest
    meshes.  The leading order is dictated by constant-panel centroid
    collocation; the second term removes the clearly resolved pre-asymptotic
    curvature without claiming a second-order BEM.  Removing that second term
    provides the main uncertainty estimate.  For a small gap,
    ``diagnostics.qd_distance_over_max_edge`` must additionally be inspected:
    extrapolation cannot compensate for an unresolved near field.
    """

    if len(subdivision_levels) < 2:
        raise ValueError("At least two subdivision_levels are required.")
    if any(
        isinstance(level, (bool, np.bool_))
        or not isinstance(level, (int, np.integer))
        for level in subdivision_levels
    ):
        raise ValueError("subdivision_levels must be strictly increasing integers >= 0.")
    levels = tuple(int(level) for level in subdivision_levels)
    if tuple(sorted(set(levels))) != levels or any(level < 0 for level in levels):
        raise ValueError("subdivision_levels must be strictly increasing integers >= 0.")
    if not np.isfinite(assumed_order) or assumed_order <= 0.0:
        raise ValueError("assumed_order must be finite and positive.")
    if (
        isinstance(correction_terms, (bool, np.bool_))
        or not isinstance(correction_terms, (int, np.integer))
        or correction_terms < 1
    ):
        raise ValueError("correction_terms must be an integer >= 1.")
    correction_count = int(correction_terms)
    minimum_fit_levels = correction_count + 1
    if extrapolation_level_count is None:
        fit_level_count = minimum_fit_levels
    else:
        if (
            isinstance(extrapolation_level_count, (bool, np.bool_))
            or not isinstance(extrapolation_level_count, (int, np.integer))
        ):
            raise ValueError("extrapolation_level_count must be an integer or None.")
        fit_level_count = int(extrapolation_level_count)
    if fit_level_count < minimum_fit_levels or fit_level_count > len(levels):
        raise ValueError(
            "extrapolation_level_count must be between correction_terms + 1 "
            "and the number of subdivision levels."
        )

    responses = tuple(
        solve_spheroid_bem(
            a_au=a_au,
            c_au=c_au,
            qd_position_au=qd_position_au,
            polarization=polarization,
            eps_m=eps_m,
            epsilon_particle=epsilon_particle,
            subdivision_level=level,
            block_size=block_size,
            max_dense_panels=max_dense_panels,
        )
        for level in levels
    )
    h = np.asarray([response.mesh.max_edge_au for response in responses], dtype=float)
    values = np.asarray(
        [_observable_array(response.observables) for response in responses],
        dtype=complex,
    )
    fit_h = h[-fit_level_count:]
    fit_values = values[-fit_level_count:]
    h_scale = float(fit_h[0])
    scaled_fit_h = fit_h / h_scale
    correction_orders = tuple(
        float(assumed_order) + float(index) for index in range(correction_count)
    )
    design = np.column_stack(
        [np.ones(fit_h.size)]
        + [scaled_fit_h**order for order in correction_orders]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, fit_values, rcond=None)
    extrapolated_values = coefficients[0]
    fitted = design @ coefficients
    absolute_fit_residual = np.max(np.abs(fit_values - fitted), axis=0)
    fit_residual = np.linalg.norm(fit_values - fitted, axis=0) / np.maximum(
        np.linalg.norm(fit_values, axis=0),
        np.finfo(float).tiny,
    )

    if correction_count == 1:
        lower_order_values = fit_values[-1]
    else:
        lower_design = design[:, :-1]
        lower_coefficients, _, _, _ = np.linalg.lstsq(
            lower_design,
            fit_values,
            rcond=None,
        )
        lower_order_values = lower_coefficients[0]
    model_stability = np.abs(extrapolated_values - lower_order_values)
    absolute_uncertainty = np.maximum(model_stability, absolute_fit_residual)
    relative_uncertainty = absolute_uncertainty / np.maximum(
        np.abs(extrapolated_values),
        np.finfo(float).tiny,
    )
    finest_change = np.abs(values[-1] - values[-2]) / np.maximum.reduce(
        [
            np.abs(values[-1]),
            np.abs(values[-2]),
            np.full(4, np.finfo(float).tiny),
        ]
    )
    return BEMConvergenceResult(
        levels=levels,
        responses=responses,
        assumed_order=float(assumed_order),
        correction_orders=correction_orders,
        extrapolation_levels=levels[-fit_level_count:],
        extrapolation_h_scale_au=h_scale,
        extrapolated=_observables_from_array(extrapolated_values),
        lower_order_extrapolated=_observables_from_array(lower_order_values),
        finest_relative_change=_errors_from_array(finest_change),
        extrapolation_relative_residual=_errors_from_array(fit_residual),
        estimated_absolute_uncertainty=_errors_from_array(absolute_uncertainty),
        estimated_relative_uncertainty=_errors_from_array(relative_uncertainty),
    )


__all__ = [
    "BEMConvergenceResult",
    "BEMDiagnostics",
    "BEMObservableErrors",
    "BEMObservables",
    "BEMResponse",
    "MAX_DENSE_PANELS",
    "TriangularSurfaceMesh",
    "build_affine_icosphere",
    "mesh_sha256",
    "run_nested_bem_convergence",
    "solve_spheroid_bem",
    "solve_spheroid_bem_on_mesh",
]
