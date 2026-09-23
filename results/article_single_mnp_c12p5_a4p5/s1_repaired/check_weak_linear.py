"""Independent frequency-domain check of the original weak N12 pulse trace."""
from qdmnp.pipeline import sha, material_fit_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
from qdmnp.observables import calculate_work_loss_spectrum_material_comparison as calc
from qdmnp.spheroid_green import qd_linear_polarizability_from_params, solve_linear_hybrid_response

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
command = next(v["command"] for k, v in manifest["steps"].items() if k.startswith("supp01_work_spectrum:"))
args = calc.parse_args(command[3:])
args.quiet = True
bundles = []
original_build = calc._build_channel_model

class Captured(Exception):
    pass

def capture(*positional, **kw):
    bundle = original_build(*positional, **kw)
    if kw["material_fit_modes"] == args.multi_fit_modes:
        bundles.append(bundle)
        raise Captured
    return bundle

try:
    with material_fit_cache(root / "material_fit_cache"), patch.object(calc, "_build_channel_model", capture):
        calc.calculate_payload(args, generator_path=Path(calc.__file__))
except Captured:
    pass
bundle = bundles[0]
trace = root / "s1_fixed/weak_trace_diagnostic.npz"
with np.load(trace) as archive:
    result = SimpleNamespace(**{k: np.array(archive[k]) for k in archive.files})
pulse = calc.pulse_for_fluence(args.fluence_j_cm2[0], energy_eV=args.carrier_energy_ev,
                              tau_fs=args.pulse_tau_fs, tau_kind=args.pulse_tau_kind, eps_m=args.eps_m)
energy = np.unique(np.concatenate((np.linspace(1.9, 2.18, 401), np.linspace(2.03, 2.055, 251))))
response = bundle.full_model.frequency_response_from_fit(energy)
beta = qd_linear_polarizability_from_params(bundle.params, energy)
reference = solve_linear_hybrid_response(response, beta, eps_m=args.eps_m).alpha_effective_au3
numerical, *_ = calc.spectral_effective_alpha_grid(result, pulse, args.eps_m, energy,
                                                   minimum_incident_relative_amplitude=args.minimum_incident_relative_amplitude)
report = {
    "scope": "weak-pulse Fourier response versus independent linear frequency-domain solution, same N12 fit",
    "trace_sha256": sha(trace), "energy_points": int(energy.size),
    "alpha_max_normalized_difference": float(np.max(abs(numerical-reference))/np.max(abs(reference))),
    "alpha_normalized_rms_difference": float(np.linalg.norm(numerical-reference)/np.linalg.norm(reference)),
    "imag_alpha_max_normalized_difference": float(np.max(abs(numerical.imag-reference.imag))/np.max(abs(reference.imag))),
}
print(json.dumps(report, indent=2), flush=True)
(Path(__file__).parent / "weak_linear_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
