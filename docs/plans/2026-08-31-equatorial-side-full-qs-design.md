# Equatorial side full-QS support

Date: 2026-08-31  
Status: approved

## Objective

Extend the quasistatic QD-MNP model from a quantum dot constrained to the
symmetry axis of a prolate spheroid to a quantum dot on its equatorial side.
The extension must retain the complete spheroidal multipole response rather
than replacing it with a point-dipole geometry factor.

The first supported side configurations use the spheroid major axis along
`z` and the QD centre at `(R, 0, 0)`:

1. `side + long`: field and QD dipole along `z`;
2. `side + trans + radial`: field and QD dipole along `x`, parallel to `R`;
3. `side + trans + tangential`: field and QD dipole along `y`.

Existing axial `long` and `trans` behaviour and serialized outputs remain
backward compatible.

## Public configuration

The existing `orientation = "long" | "trans"` continues to select the bright
MNP polarizability. Two independent geometry fields are added:

```python
qd_placement = "axis" | "side"
side_transverse_alignment = None | "radial" | "tangential"
```

`side_transverse_alignment` is mandatory for `side + trans` and invalid in
all other configurations. The scalar `R` remains the centre-to-centre
distance. The directional surface gap is

\[
g=R-c-r_{\rm QD}\quad\text{(axis)},\qquad
g=R-a-r_{\rm QD}\quad\text{(side)}.
\]

The far-field geometry factors are retained only as dipole-limit diagnostics:

| placement | polarization | far-field `G` | bright factor |
|---|---|---:|---|
| axis | long | 2 | \(L_{10}\) |
| axis | trans | -1 | \(L_{11}\) |
| side | long | -1 | \(L_{10}\) |
| side | trans radial | 2 | \(L_{11}\) |
| side | trans tangential | -1 | \(L_{11}\) |

## Architecture

The validated axial `ProlateSpheroidGeometry`, `SpheroidGreenInteraction`, and
`QuasistaticInteractionResponse` remain intact. A sibling equatorial geometry,
kernel, and by-mode response are added. Both kernels expose a common internal
modal contract containing:

```text
mode_degree
mode_order
mode_sector
depolarization
reaction_weight
bright_mode_index
bright_source_coupling
n_max
mode_count
```

The equatorial response stores values by `(n,m,sector)` and additionally
aggregates `K` by complete degree shells. Truncation and convergence therefore
continue to mean truncation in the maximum spatial degree, not in a flattened
mode count.

The external scalar port is unchanged:

\[
p_M=A E_{\rm inc}+B\mu_D,
\qquad
E_{M\to D}=B E_{\rm inc}+K\mu_D.
\]

## Equatorial coordinates and modes

For semiaxes `a <= c`,

\[
f=\sqrt{c^2-a^2},\quad
\rho=\sqrt{R^2+f^2},\quad
\xi_0=c/f,\quad
\xi_D=\rho/f,\quad
\eta_D=0,\quad\phi_D=0.
\]

The real exterior basis is

\[
U_{nm}^{c/s}=\mathsf Q_n^m(\xi)P_n^m(\eta)
\{\cos(m\phi),\sin(m\phi)\}.
\]

For every active mode,

\[
L_{nm}=\frac{r^P_{nm}}{r^P_{nm}+r^Q_{nm}},\quad
r^P_{nm}=\frac{\partial_\xi\mathsf P_n^m}{\mathsf P_n^m},\quad
r^Q_{nm}=-\frac{\partial_\xi\mathsf Q_n^m}{\mathsf Q_n^m},
\]

\[
\chi_{nm}(\omega)=
\frac{\varepsilon_p-\varepsilon_m}
{\varepsilon_m+L_{nm}(\varepsilon_p-\varepsilon_m)}.
\]

With

\[
H_{nm}=(2n+1)(2-\delta_{m0})(-1)^m
\left[\frac{(n-m)!}{(n+m)!}\right]^2,
\]

\[
g_{nm}=-L_{nm}\frac{\mathsf P_n^m(\xi_0)}
{\mathsf Q_n^m(\xi_0)},
\]

the reaction weight is

\[
w_j=-\frac{H_{nm}g_{nm}}{\varepsilon_m f}D_j^2>0.
\]

The channel-specific derivatives and selection rules are:

| channel | sector | selection | \(D_j\) |
|---|---|---|---|
| side long (`z`) | cosine | `n+m` odd | \(Q_n^m(\xi_D)[P_n^m]'(0)/\rho\) |
| side radial (`x`) | cosine | `n+m` even | \(R[Q_n^m]'(\xi_D)P_n^m(0)/(f\rho)\) |
| side tangential (`y`) | sine | `m>=1`, `n+m` even | \(mQ_n^m(\xi_D)P_n^m(0)/R\) |

The induced monopole is omitted. The bright modes are `(1,0,cos)` for `long`,
`(1,1,cos)` for radial transverse, and `(1,1,sin)` for tangential transverse.

## Bright port and exact identities

With

\[
C=\varepsilon_m a^2c/3,
\]

the bright response is

\[
A=C\chi_b,\qquad B=\lambda A,\qquad
K=\sum_j w_j\chi_j.
\]

Define

\[
t=f/\rho,\qquad
F(t)=\frac{\operatorname{atanh}t-t}{t^3}.
\]

Near zero, `F` is evaluated from

\[
F(t)=1/3+t^2/5+t^4/7+\cdots.
\]

The stable bright couplings are

\[
\lambda_z=-\frac{3F(t)}{\varepsilon_m\rho^3},
\]

\[
\lambda_x=\frac{3}{2\varepsilon_m\rho^3}
\left[F(t)+\frac{1}{1-t^2}\right],
\]

\[
\lambda_y=-\frac{3}{2\varepsilon_m\rho^3}
\left[\frac{1}{1-t^2}-F(t)\right].
\]

Every implementation must satisfy

\[
w_b=C\lambda^2,\qquad K_b=B^2/A,
\qquad \lambda_x+\lambda_y=-\lambda_z.
\]

Their spherical and far-field limits recover `G = -1, 2, -1`.

## Numerical representation

Associated Legendre functions are combined in normalized signed-log form.
Angular values at zero use analytic double-factorial formulas. SciPy radial
functions are guarded by finite-value and Wronskian checks. The first supported
equatorial maximum is `n_max <= 80`; the existing axial high-order guard is not
silently reused.

An exact spherical branch aggregates degenerate `m` contributions by degree.
Spatial convergence compares `N` with `floor(N/2)` and also measures the
absolute contribution of the final complete degree shells. This prevents
cancellation across `m` from producing a false convergence certificate.

## Time-domain models

The direct reference model retains every active spatial mode and applies the
existing passive material transform

\[
\chi_j=\frac{H_b}{1+(L_j-L_b)H_b}.
\]

It is intended for low spatial order only.

The production model leaves the bright mode exact and reduces only

\[
K_{\rm dark}=\sum_{j\ne b}w_j
\frac{H_b}{1+(L_j-L_b)H_b}.
\]

The positive spectral measure

\[
d\nu(L)=\sum_{j\ne b}w_j\delta(L-L_j)dL
\]

is clustered adaptively into positive nodes `(L_hat_q, W_q)`. Positive weights
and `0 < L_hat_q < 1` preserve passivity. Fit and audit frequency grids are
different. Default acceptance limits are normalized RMS error `<= 1e-6` and
maximum normalized error `<= 1e-4`. A failed reduction is an error by default.

## Independent BEM validation

The validation oracle is a Cartesian three-dimensional apparent-surface-charge
BEM on an affine-mapped icosphere mesh. It must not import or use spheroidal
functions, `L_nm`, or analytic reaction weights.

It solves the dielectric second-kind boundary integral equation for a uniform
field and independently for a point-dipole source. From the two right-hand
sides it extracts `A`, `B_field`, `B_dipole`, and `K`. Equality of the two `B`
values is an independent reciprocity check.

Offline validation uses nested meshes, adaptive near-panel quadrature, local
refinement near small gaps, and extrapolation to zero mesh size. CI runs fast
sphere/convergence checks and compares the analytic kernel against immutable
high-resolution fixtures with mesh and quadrature provenance.

Implementation-scope note (2026-09-02): the checked-in independent oracle
currently uses piecewise-constant centroid collocation on globally refined
affine-icosphere meshes plus two-term zero-mesh extrapolation.  Adaptive
near-panel quadrature and local refinement were not implemented in this
iteration.  The resulting immutable fixture therefore certifies only its
listed discrete cases, including the smallest tested `gap/a = 0.5`; it makes
no validation claim for `gap/a < 0.5`.  Extending that range requires the
adaptive/local methods above and a regenerated independent fixture.

Acceptance targets are 0.5% for ordinary `A/B`, 1% for ordinary `K`, and 2%
for small-gap `K`, while also lying within three estimated BEM uncertainties.
Failure of BEM self-convergence never widens the analytic-kernel tolerance.

## Required verification

- all previous axial tests remain bitwise or tightly numerically compatible;
- zero dielectric contrast;
- length scaling of `A/B/K`;
- mode selection and shell aggregation;
- reciprocity and nonnegative passive modal loss;
- exact sphere and near-sphere rotational covariance;
- far-field point-dipole limits for all three side channels;
- frequency/direct-time equivalence at low order;
- reduced/full frequency and time equivalence;
- coupled ground-state stability, work passivity, and density-matrix bounds;
- independent BEM fixtures for sphere, axial, and all side channels;
- CLI metadata distinguishes `n_max`, full mode count, and reduced node count.

## Delivery sequence

1. Geometry/API and point-dipole baseline.
2. Exact equatorial frequency-domain modal kernel.
3. Low-order direct time-domain integration.
4. Positive spectral-measure reduction and production time model.
5. Independent BEM solver, converged fixtures, and comparison reports.
6. CLI integration, documentation, full regression, and performance audit.
