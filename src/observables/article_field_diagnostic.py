"""Bare-spheroid longitudinal fields from an independent ellipsoidal potential.

This inexpensive article diagnostic needs physical inputs only. It does not fit
the metal, integrate QD dynamics, or load earlier calculation archives. The QD
radius and surface gap locate an observation point, with no QD back-action.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from qdmnp.rational_fit import DEFAULT_AU_MATERIAL


def _geometry(c_nm, a_nm, eps_m):
    if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0
           for v in (c_nm, a_nm, eps_m)):
        raise ValueError("Semi-axes and medium permittivity must be finite and positive.")
    if a_nm > c_nm:
        raise ValueError("Require a_nm <= c_nm for a prolate spheroid or sphere.")


def _potential_integral(axis_ratio_squared, lower):
    """Dimensionless confocal integral; a bounded change of variable avoids units.

    I(lambda) = integral_lambda^infinity ds / [(q+s)*(1+s)^(3/2)],
    q=(a/c)^2. Substitution t=1/sqrt(1+s) maps it to a finite interval.
    """
    upper = 1 / np.sqrt(1 + lower)
    return quad(lambda t: 2*t*t/(1-(1-axis_ratio_squared)*t*t),
                0, upper, epsabs=1e-13, epsrel=1e-12)[0]


def uniform_field_ratio(c_nm, a_nm, distance_from_surface_nm, epsilon, eps_m, placement):
    """Exact exterior Ez/Einc of a bare spheroid with incident field along c.

    ``placement`` is ``axis`` (tip) or ``side`` (equator). Distance is measured
    from the metal surface, including the exterior one-sided surface limit at
    zero. ``epsilon`` can be a scalar or array; the distance is scalar.
    This is the confocal-potential formula used in the independent historical
    tip/side audit, evaluated in dimensionless coordinates rather than nm.
    """
    _geometry(c_nm, a_nm, eps_m)
    if placement not in ("axis", "side"):
        raise ValueError("placement must be axis or side.")
    if (isinstance(distance_from_surface_nm, bool)
            or not np.isfinite(distance_from_surface_nm)
            or distance_from_surface_nm < 0):
        raise ValueError("Distance from the metal surface must be finite and nonnegative.")
    epsilon = np.asarray(epsilon, dtype=complex)
    if not np.all(np.isfinite(epsilon)):
        raise ValueError("Particle permittivity must be finite.")
    q = (a_nm/c_nm)**2
    Lz = q/2 * _potential_integral(q, 0)
    chi = (epsilon-eps_m)/(eps_m+Lz*(epsilon-eps_m))
    radius = ((c_nm if placement == "axis" else a_nm)
              + distance_from_surface_nm)/c_nm
    lower = radius*radius - (1 if placement == "axis" else q)
    integral = _potential_integral(q, max(0., lower))
    derivative = 2/((q+lower)*radius)-integral if placement == "axis" else -integral
    return 1 + q/2*chi*derivative


def build_field_diagnostic(c_nm, a_nm, eps_m, carrier_energy_eV, qd_radius_nm, gap_nm,
                           *, energy_min_eV=1.6, energy_max_eV=2.2,
                           energy_points=1201, distance_max_nm=10., distance_points=1001):
    """Return fresh plot arrays and a JSON-ready interpretation of bare fields.

    Frequency profiles vary the material response at a fixed observation point;
    they do not retune the QD transition or predict a nonlinear excitation threshold.
    """
    _geometry(c_nm, a_nm, eps_m)
    for name, value, positive in (("carrier energy", carrier_energy_eV, True),
                                  ("QD radius", qd_radius_nm, True),
                                  ("gap", gap_nm, False),
                                  ("maximum distance", distance_max_nm, True)):
        if (isinstance(value, bool) or not np.isfinite(value)
                or value < 0 or (positive and value == 0)):
            raise ValueError(f"Invalid {name}.")
    if not (np.isfinite(energy_min_eV) and np.isfinite(energy_max_eV)
            and 0 < energy_min_eV <= carrier_energy_eV <= energy_max_eV
            and energy_min_eV < energy_max_eV):
        raise ValueError("The positive energy interval must contain the carrier.")
    for points in (energy_points, distance_points):
        if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or points < 2:
            raise ValueError("Plot grids need at least two points.")
    distance = qd_radius_nm + gap_nm
    # Include the actual observation point even for configurations beyond 10 nm.
    distances = np.linspace(0., max(distance_max_nm, distance), distance_points)
    energies = np.linspace(energy_min_eV, energy_max_eV, energy_points)
    epsilon_grid = DEFAULT_AU_MATERIAL.epsilon_at(energies)
    epsilon = complex(DEFAULT_AU_MATERIAL.epsilon_at(carrier_energy_eV))
    channels, spectrum, profile, centre, surface = {}, [], [], {}, {}
    for placement in ("axis", "side"):
        ratio = complex(uniform_field_ratio(c_nm, a_nm, distance, epsilon, eps_m, placement))
        spectral_ratio = uniform_field_ratio(c_nm, a_nm, distance, epsilon_grid, eps_m, placement)
        spatial_ratio = np.asarray([uniform_field_ratio(c_nm, a_nm, d, epsilon, eps_m, placement)
                                    for d in distances])
        spectrum.append(abs(spectral_ratio)**2)
        profile.append(abs(spatial_ratio)**2)
        centre[placement], surface[placement] = float(abs(ratio)**2), float(abs(spatial_ratio[0])**2)
        channels[placement] = {
            "field_ratio_real": ratio.real, "field_ratio_imag": ratio.imag,
            "scattered_field_intensity_gain": float(abs(ratio-1)**2),
            "bare_field_intensity_gain": centre[placement],
        }
    profile, spectrum = np.asarray(profile), np.asarray(spectrum)
    difference = profile[0]-profile[1]
    crossing = None
    for index in range(len(distances)-1):
        if difference[index] == 0:
            crossing = float(distances[index])
            break
        if difference[index]*difference[index+1] < 0:
            crossing = float(brentq(lambda d: float(abs(uniform_field_ratio(c_nm, a_nm, d, epsilon, eps_m, "axis"))**2
                                                   - abs(uniform_field_ratio(c_nm, a_nm, d, epsilon, eps_m, "side"))**2),
                                    distances[index], distances[index+1], xtol=1e-11))
            break
    q = (a_nm/c_nm)**2
    Lz = q/2 * _potential_integral(q, 0)
    interpretation = {
        "scope": "Bare MNP in a uniform longitudinal external field; no QD back-action, transition retuning, or threshold prediction",
        "c_nm": float(c_nm), "a_nm": float(a_nm), "eps_m": float(eps_m),
        "carrier_energy_eV": float(carrier_energy_eV),
        "qd_radius_nm": float(qd_radius_nm), "gap_nm": float(gap_nm),
        "centre_distance_from_surface_nm": float(distance), "Lz": float(Lz),
        "epsilon_real": epsilon.real, "epsilon_imag": epsilon.imag,
        "centre_field_gain": centre, "surface_field_gain": surface,
        "first_profile_crossing_nm": crossing,
        "profile_distance_interval_nm": [float(distances[0]), float(distances[-1])],
        "channels": channels,
    }
    return {"energy_eV": energies, "distance_from_surface_nm": distances,
            "placements": ("axis", "side"), "spectrum_field_gain": spectrum,
            "profile_field_gain": profile, "interpretation": interpretation}


def plot_field_diagnostic(data, output, *, dpi=200):
    """Render the two fresh diagnostic arrays; write only the requested figure."""
    import matplotlib.pyplot as plt

    output = Path(output)
    info = data["interpretation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for index, label in enumerate(("tip", "side, E parallel to c")):
        axes[0].semilogy(data["energy_eV"], data["spectrum_field_gain"][index], label=label)
        axes[1].semilogy(data["distance_from_surface_nm"], data["profile_field_gain"][index], label=label)
    axes[0].axvline(info["carrier_energy_eV"], color="black", ls=":", label="fixed carrier")
    axes[1].axvline(info["centre_distance_from_surface_nm"], color="black", ls=":", label="QD centre")
    axes[0].set(xlabel="Photon energy (eV)", ylabel="Bare-MNP total field |E/Einc|^2")
    axes[1].set(xlabel="Distance of observation point from surface (nm)",
                ylabel="|E/Einc|^2 at fixed carrier")
    for ax in axes:
        ax.grid(True, alpha=.25)
        ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("c-nm", "a-nm", "eps-m", "carrier-energy-ev", "qd-radius-nm", "gap-nm"):
        parser.add_argument("--"+name, type=float, required=True)
    parser.add_argument("--energy-min-ev", type=float, default=1.6)
    parser.add_argument("--energy-max-ev", type=float, default=2.2)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = build_field_diagnostic(args.c_nm, args.a_nm, args.eps_m, args.carrier_energy_ev,
                                  args.qd_radius_nm, args.gap_nm,
                                  energy_min_eV=args.energy_min_ev, energy_max_eV=args.energy_max_ev)
    return plot_field_diagnostic(data, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
