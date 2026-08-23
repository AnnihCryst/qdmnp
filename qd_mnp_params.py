"""Общие переопределения физических параметров для скриптов КТ-МНЧ.

Базовый набор параметров задается функцией ``make_default_params()`` в
``qd_mnp_rational_fit.py``. Этот модуль заменяет только те величины,
которые явно переданы из командной строки, и переводит удобные физические
единицы (нм, эВ, мэВ, Дебай) в атомные единицы решателя.

Модуль импортируется из ``qd_mnp_linear_spectrum.py``, ``qd_mnp_fano_scan.py``
и ``qd_mnp_pulse_absorption_sweep.py``. Запускать его напрямую не нужно.
"""

from __future__ import annotations

from dataclasses import replace

from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    eV_to_au,
    nm_to_au,
    make_default_params,
)


DEBYE_C_M = 3.33564e-30


def make_params_with_overrides(
    *,
    c_nm: float | None = None,
    a_nm: float | None = None,
    r_nm: float | None = None,
    g_factor: float | None = None,
    eps_m: float | None = None,
    d_debye: float | None = None,
    omega0_ev: float | None = None,
    gamma_population_mev: float | None = None,
    gamma_dephasing_mev: float | None = None,
):
    """Вернуть параметры модели с необязательной заменой физических величин.

    Значения None оставляют параметры по умолчанию. c/a - полуоси эллипсоида
    МНЧ, R - расстояние между центрами КТ и МНЧ, G - геометрическая или
    эффективная сила диполь-дипольной связи, eps_m - проницаемость среды,
    d - переходный диполь КТ, omega0 - энергия перехода, gamma - релаксация
    населенности, Gamma - дефазировка когерентности.
    """
    params = make_default_params()
    updates = {}
    if c_nm is not None:
        updates["c_au"] = float(nm_to_au(c_nm))
    if a_nm is not None:
        updates["a_au"] = float(nm_to_au(a_nm))
    if r_nm is not None:
        updates["R_au"] = float(nm_to_au(r_nm))
    if g_factor is not None:
        updates["G"] = float(g_factor)
    if eps_m is not None:
        updates["eps_m"] = float(eps_m)
    if d_debye is not None:
        updates["d_au"] = float(d_debye * DEBYE_C_M / AU_DIPOLE_C_M)
    if omega0_ev is not None:
        updates["omega0_au"] = float(eV_to_au(omega0_ev))
    if gamma_population_mev is not None:
        updates["gamma_au"] = float(eV_to_au(gamma_population_mev / 1000.0))
    if gamma_dephasing_mev is not None:
        updates["Gamma_au"] = float(eV_to_au(gamma_dephasing_mev / 1000.0))
    return replace(params, **updates)
