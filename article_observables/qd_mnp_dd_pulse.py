"""Apply the article time-resolution request to the unchanged DD solver."""

from __future__ import annotations

import numpy as np


def solve_dd_with_resolution(model, pulse, *, points_per_fastest_cycle, **solve_kwargs):
    """Resolve incident, material, coupled and observed nonlinear frequencies.

    The DD core enforces its own minimum of 20 points per cycle.  This adapter
    additionally enforces the article request, including an increased local
    Rabi frequency discovered by that core.  At most three reruns are allowed.
    """
    points = float(points_per_fastest_cycle)
    if not np.isfinite(points) or points <= 0.0:
        raise ValueError("points_per_fastest_cycle must be finite and positive.")
    requested_step = solve_kwargs.pop("max_step_au", None)
    if requested_step is not None:
        requested_step = float(requested_step)
        if not np.isfinite(requested_step) or requested_step <= 0.0:
            raise ValueError("max_step_au must be finite and positive.")
    frequencies = np.concatenate((
        np.asarray([
            pulse.omegaL_au,
            model.params.omega0_au,
            2.0 * abs(model.params.d_au * model.params.qd_local_field_factor * pulse.E0_au),
        ], dtype=float),
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
        result = model.solve(pulse, max_step_au=step, **solve_kwargs)
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
        if float(np.max(np.diff(times))) <= required_step * (1.0 + 1.0e-12):
            return result
        if attempt == 3:
            raise RuntimeError(
                "DD trajectory does not resolve the observed frequency ceiling "
                "at the requested points_per_fastest_cycle after three reruns."
            )
    raise AssertionError("Unreachable DD resolution loop exit.")
