"""Apply the article time-resolution request to the unchanged DD solver."""

from __future__ import annotations

import numpy as np


def solve_dd_with_resolution(model, pulse, *, points_per_fastest_cycle, step_frequency_policy="all_poles", **solve_kwargs):
    """Resolve incident, material, coupled and observed nonlinear frequencies.

    The same points-per-cycle request and frequency policy are passed to the DD
    core, including an increased local Rabi frequency discovered by that core.
    ``excited_band`` resolves carrier, exciton and Rabi frequencies only; the
    material poles are then controlled by rtol/atol. At most three reruns.
    """
    points = float(points_per_fastest_cycle)
    if not np.isfinite(points) or points <= 0.0:
        raise ValueError("points_per_fastest_cycle must be finite and positive.")
    requested_step = solve_kwargs.pop("max_step_au", None)
    if requested_step is not None:
        requested_step = float(requested_step)
        if not np.isfinite(requested_step) or requested_step <= 0.0:
            raise ValueError("max_step_au must be finite and positive.")
    if step_frequency_policy not in ("all_poles", "excited_band"):
        raise ValueError("step_frequency_policy must be 'all_poles' or 'excited_band'.")
    band = np.asarray([
        pulse.omegaL_au,
        model.params.omega0_au,
        2.0 * abs(model.params.d_au * model.params.qd_local_field_factor * pulse.E0_au),
    ], dtype=float)
    frequencies = band if step_frequency_policy == "excited_band" else np.concatenate((
        band,
        np.abs(np.asarray(model.fit.omega_modes_au)).reshape(-1),
        np.abs(np.asarray(model.linear_stability.poles_au)).reshape(-1),
    ))
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("DD resolution frequencies must be finite and nonnegative.")
    ceiling = float(np.max(frequencies))
    if ceiling <= 0.0:
        raise ValueError("DD resolution needs a positive frequency ceiling.")
    for attempt in range(4):
        step = 2.0 * np.pi / (points * ceiling)
        if requested_step is not None:
            step = min(step, requested_step)
        # Legacy all_poles keeps the core's own 20-point minimum unchanged.
        core_points = max(points, 20.0) if step_frequency_policy == "all_poles" else points
        result = model.solve(pulse, max_step_au=step, points_per_fastest_cycle=core_points,
                             step_frequency_policy=step_frequency_policy, **solve_kwargs)
        observed = float(result.diagnostics.integration_frequency_ceiling_au)
        if not np.isfinite(observed) or observed <= 0.0:
            raise RuntimeError("DD solver returned an invalid frequency ceiling.")
        ceiling = max(ceiling, observed)
        required_step = 2.0 * np.pi / (points * ceiling)
        if requested_step is not None:
            required_step = min(required_step, requested_step)
        times = np.asarray(result.t_au, dtype=float)
        if (times.ndim != 1 or times.size < 2 or np.any(~np.isfinite(times))
                or np.any(np.diff(times) <= 0.0)):
            raise RuntimeError("DD solver returned an invalid time grid.")
        # t_new - t is rounded in floating point: at t ~ 1e5 a.u. the excess over
        # max_step reaches ~1e-11 of the step, which is not an unresolved cycle.
        rounding = 8.0 * float(np.spacing(max(abs(times[0]), abs(times[-1]))))
        if float(np.max(np.diff(times))) <= required_step * (1.0 + 1.0e-12) + rounding:
            return result
        if attempt == 3:
            raise RuntimeError(
                "DD trajectory does not resolve the observed frequency ceiling "
                "at the requested points_per_fastest_cycle after three reruns."
            )
    raise AssertionError("Unreachable DD resolution loop exit.")
