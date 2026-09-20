"""Deterministic positive-Lorentz fitting helpers (no dynamical-model changes).

All frequencies, dampings and strengths use eV, eV and eV**2 here. The caller
converts to atomic units once. Auxiliary UV poles are a numerical realization
of a finite-band background, not measured resonances or an extrapolation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares, nnls


@dataclass(frozen=True)
class PassiveFitRefinement:
    focus_center_eV: float
    focus_half_width_eV: float = .0075
    focus_relative_error: float = .0025
    pole_bound_factor: float = 4.

    def __post_init__(self):
        values = (self.focus_center_eV, self.focus_half_width_eV,
                  self.focus_relative_error, self.pole_bound_factor)
        if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Passive-fit refinement values must be finite and positive.')
        if self.focus_relative_error >= 1 or self.pole_bound_factor < 1:
            raise ValueError('Require focus_relative_error < 1 and pole_bound_factor >= 1.')


def lorentz_values_jacobian(energy, u, alpha_inf):
    n = len(u) // 3
    f, w, g = u[:n], np.exp(u[n:2*n]), np.exp(u[2*n:])
    denominator = w**2 - energy[:, None]**2 - 1j*energy[:, None]*g
    values = alpha_inf + np.sum(f/denominator, axis=1)
    jacobian = np.concatenate((1/denominator,
        -2*f*w*w/denominator**2, 1j*f*g*energy[:, None]/denominator**2), axis=1)
    return values, jacobian


def positive_lorentz_candidate(energy, target, alpha_inf, n_modes, *,
        omega_bounds, gamma_bounds, strength_max, alpha_weight=1., inverse_weight=1.2,
        nrms_limit=.025, pointwise_limit=.05, focus_center=None, focus_half_width=None,
        focus_relative_error=.0025):
    """Return passive coefficients; caller must independently enforce its gates.

    Positive matching pursuit + NNLS avoids nearly zero-strength inserted poles,
    whose frequency/damping derivatives otherwise vanish. Residual-directed pole
    relocation escapes poor local minima. A final feasibility objective targets
    the actual global error gates, optionally with local complex-response accuracy.
    alpha_inf is fixed exactly and is never an optimization coordinate.
    """
    energy = np.asarray(energy, dtype=float)
    target = np.asarray(target, dtype=complex)
    # Optimize starting guesses on a bounded grid; certify/polish on ALL caller
    # points below. Interpolation does not add independent material information.
    indices = np.unique(np.linspace(0, len(energy)-1, min(len(energy), 1100)).astype(int))
    train_e, train_a = energy[indices], target[indices]
    scales = [max(np.max(abs(v)), 1e-12) for v in
              (train_a.real, train_a.imag, (1/train_a).real, (1/train_a).imag)]
    factors = (np.sqrt(alpha_weight), np.sqrt(inverse_weight))
    n = n_modes
    lo = np.r_[np.zeros(n), np.full(n, np.log(omega_bounds[0])), np.full(n, np.log(gamma_bounds[0]))]
    hi = np.r_[np.full(n, strength_max), np.full(n, np.log(omega_bounds[1])), np.full(n, np.log(gamma_bounds[1]))]

    def objective(u):
        value, j = lorentz_values_jacobian(train_e, u, alpha_inf)
        delta, inverse_delta = value-train_a, 1/value-1/train_a
        ji = -j/value[:, None]**2
        r = np.r_[factors[0]*delta.real/scales[0], factors[0]*delta.imag/scales[1],
                  factors[1]*inverse_delta.real/scales[2], factors[1]*inverse_delta.imag/scales[3]]
        jac = np.vstack((factors[0]*j.real/scales[0], factors[0]*j.imag/scales[1],
                         factors[1]*ji.real/scales[2], factors[1]*ji.imag/scales[3]))
        return r, jac

    def polish(u, iterations):
        return least_squares(lambda x: objective(x)[0], np.clip(u, lo+1e-12, hi-1e-12),
            jac=lambda x: objective(x)[1], bounds=(lo, hi), x_scale='jac',
            max_nfev=iterations, ftol=2e-10, xtol=2e-10, gtol=2e-10).x

    # Include broad UV background terms as well as resolved in-band resonances.
    ws = np.unique(np.r_[np.linspace(max(.3, omega_bounds[0]*1.02), min(4.5,omega_bounds[1]/1.02),101),
                         np.array([5.,6.,8.,12.,16.])])
    ws = ws[(ws > omega_bounds[0]) & (ws < omega_bounds[1])]
    gs = np.array([.02,.04,.08,.15,.3,.6,1.2,2.4,4.8,8.])
    gs = np.unique(np.clip(gs, gamma_bounds[0]*1.02, gamma_bounds[1]/1.02))
    ws, gs = (v.ravel() for v in np.meshgrid(ws,gs))
    dictionary = 1/(ws**2-train_e[:,None]**2-1j*train_e[:,None]*gs)
    def design(reference):
        inv = -dictionary/reference[:,None]**2
        return np.vstack((factors[0]*dictionary.real/scales[0], factors[0]*dictionary.imag/scales[1],
                          factors[1]*inv.real/scales[2], factors[1]*inv.imag/scales[3]))
    matrix = design(train_a)
    rhs = np.r_[factors[0]*(train_a-alpha_inf).real/scales[0], factors[0]*train_a.imag/scales[1],
                factors[1]*(-(train_a-alpha_inf)/train_a**2).real/scales[2],
                factors[1]*(-(train_a-alpha_inf)/train_a**2).imag/scales[3]]
    norms = np.linalg.norm(matrix, axis=0)
    matrix /= norms
    selected, remainder = [], rhs.copy()
    for _ in range(n):
        scores = matrix.T @ remainder
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
        amplitudes, _ = nnls(matrix[:,selected], rhs, maxiter=3000)
        remainder = rhs-matrix[:,selected]@amplitudes
    u = np.r_[amplitudes/norms[selected], np.log(ws[selected]), np.log(gs[selected])]
    u = polish(u,1500)
    candidates = [u.copy()]
    for _ in range(3):
        r, j = objective(u)
        impact = u[:n]*np.linalg.norm(j[:,:n],axis=0)
        replace = int(np.argmin(impact))
        value, _ = lorentz_values_jacobian(train_e,u,alpha_inf)
        basis = design(value)
        basis_norm = np.linalg.norm(basis,axis=0)
        correlation = -(basis.T@r)/basis_norm
        picks = []
        for ix in np.argsort(correlation)[::-1]:
            if all(abs(np.log(ws[ix]/ws[p]))+abs(np.log(gs[ix]/gs[p])) > .4 for p in picks):
                picks.append(int(ix))
            if len(picks) == 8: break
        best, score = u, np.linalg.norm(r)
        for ix in picks:
            trial = u.copy()
            trial[replace] = max(correlation[ix]/basis_norm[ix],1e-7)
            trial[n+replace], trial[2*n+replace] = np.log(ws[ix]), np.log(gs[ix])
            fitted = polish(trial,500)
            newscore = np.linalg.norm(objective(fitted)[0])
            if newscore < score: best, score = fitted,newscore
        if np.linalg.norm(best-u) < 1e-10: break
        u = best
        candidates.append(u.copy())

    global_norm = np.linalg.norm(target)
    inverse_target = 1/target
    inverse_norm = np.linalg.norm(inverse_target)
    focused = np.zeros(len(energy),dtype=bool) if focus_center is None else abs(energy-focus_center)<=focus_half_width
    # Local accuracy is additional numerical refinement, never a replacement
    # for the unweighted full-band gates. Use a modest margin for audit grids.
    local_limit = focus_relative_error
    rms_goal, point_goal = .99*nrms_limit, .99*pointwise_limit
    def feasibility(u, local_weight=10.):
        v,j = lorentz_values_jacobian(energy,u,alpha_inf)
        delta, di = v-target, 1/v-inverse_target
        ji = -j/v[:,None]**2
        relative = abs(delta)/abs(target)
        nr, ni = np.linalg.norm(delta)/global_norm, np.linalg.norm(di)/inverse_norm
        jr = (delta.conj()[:,None]*j).real/np.maximum(abs(delta)*abs(target),1e-100)[:,None]
        jnr = np.sum((delta.conj()[:,None]*j).real,axis=0)/max(np.linalg.norm(delta)*global_norm,1e-100)
        jni = np.sum((di.conj()[:,None]*ji).real,axis=0)/max(np.linalg.norm(di)*inverse_norm,1e-100)
        r = np.r_[np.maximum(relative-point_goal,0),100*max(nr-rms_goal,0),100*max(ni-rms_goal,0),
                  .002*delta.real/abs(target), .002*delta.imag/abs(target),
                  local_weight*np.maximum(relative[focused]-local_limit,0)]
        jac = np.vstack((jr*(relative>point_goal)[:,None],100*jnr*(nr>rms_goal),100*jni*(ni>rms_goal),
                         .002*j.real/abs(target)[:,None],.002*j.imag/abs(target)[:,None],
                         local_weight*jr[focused]*(relative[focused]>local_limit)[:,None]))
        violation = max(nr/nrms_limit,ni/nrms_limit,np.max(relative)/pointwise_limit)
        if np.any(focused): violation=max(violation,np.max(relative[focused])/(1.1*local_limit))
        return r,jac,violation
    feasible_candidates = []
    # Different basins can trade maximum error against mean-square error.
    for start in candidates[::-1]:
        global_result = least_squares(lambda x:feasibility(x,0.)[0],start,
            jac=lambda x:feasibility(x,0.)[1], bounds=(lo,hi),x_scale='jac',
            max_nfev=2000,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        result = least_squares(lambda x:feasibility(x)[0],global_result.x,jac=lambda x:feasibility(x)[1],
            bounds=(lo,hi),x_scale='jac',max_nfev=4000,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        feasible_candidates.append(result.x)
        if feasibility(result.x)[2] <= 1: break
    return min(feasible_candidates,key=lambda x:feasibility(x)[2])
