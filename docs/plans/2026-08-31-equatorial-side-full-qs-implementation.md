# Equatorial side full-QS implementation plan

## Stage 1: geometry and public configuration

- Add `qd_placement` and `side_transverse_alignment` with axial-compatible
  defaults to the native parameter model.
- Add a geometry-aware far-field interaction factor without changing the
  existing public `orientation_factor` behaviour.
- Add directional surface-gap validation and metadata.
- Extend parameter and native point-dipole tests for all five supported
  placement/polarization combinations.

## Stage 2: exact equatorial frequency kernel

- Add equatorial geometry and immutable mode metadata.
- Enumerate the symmetry-reduced cosine/sine `(n,m)` sets.
- Implement exact-sphere shell weights.
- Implement stable prolate `L_nm`, bright couplings, and reaction weights with
  signed-log scaling and numerical guards.
- Add by-mode response, by-degree aggregation, truncation, and convergence
  diagnostics.
- Verify low-order identities, sphere covariance, far-field limits,
  passivity, and no-contrast behaviour.

## Stage 3: common modal interface and direct time reference

- Adapt the full-QS time model to consume explicit modal metadata, a bright
  index, and mode count independently from `n_max`.
- Preserve the existing axial code path and outputs.
- Add low-order direct side time-domain construction and weak-field
  time/frequency regression tests for all side channels.

## Stage 4: positive dark-kernel reduction

- Implement deterministic positive clustering of the dark spectral measure in
  depolarization-factor space.
- Use independent construction and audit grids.
- Preserve the exact bright channel and expose reduction diagnostics.
- Add a production reduced time-domain backend and compare it against the full
  frequency response and low-order direct trajectories.

## Stage 5: independent Cartesian BEM

- Add deterministic affine-icosphere triangulation and a Cartesian
  apparent-surface-charge boundary-integral solver.
- Calibrate signs and normalization against exact sphere responses.
- Extract `A`, two independent reciprocal `B` values, and `K` for uniform and
  point-dipole right-hand sides.
- Add nested-mesh convergence, quadrature provenance, and an offline fixture
  generator.
- Generate and validate prolate axial and equatorial fixtures.

## Stage 6: runners, artifacts, and documentation

- Add placement/alignment flags to frequency and pulse runners while keeping
  old invocations unchanged.
- Export `(n,m,sector)`, mode count, spatial degree, reduction count, and all
  accuracy certificates.
- Document physical meaning, limitations, convergence workflow, and example
  invocations.

## Verification gates

1. Focused tests after every stage.
2. Existing axial full-QS regression suite.
3. Full repository test suite.
4. Static compile/import checks.
5. Frequency, direct-time, reduced-time, and BEM cross-checks.
6. Final dirty-worktree and diff audit.
