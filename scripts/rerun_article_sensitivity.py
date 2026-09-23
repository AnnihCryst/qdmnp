"""Recheck failed article sensitivity cases without replacing the original run.

Reuse its physical inputs and acceptance limits, refine spatial order, and
recompute a combined ranking verdict with explicit provenance for every case.
This is a sensitivity supplement, never a rewrite of the original manifest.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from qdmnp.pipeline import (
    ROOT, PREFIX, ArticleRun, assess_recommendation_ranking, compare_thresholds,
    material_fit_cache, read_npz, sha, threshold_subset, write_json,
)
import numpy as np
from qdmnp.rational_fit import AU_ENERGY_EV


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=("gap_low", "c_low", "a_high"),
                        default=["gap_low", "c_low", "a_high"])
    parser.add_argument("--spatial-order", type=int, default=160)
    parser.add_argument("--material-modes", type=int, default=13,
                        help="Pole count for the changed shapes; nominal gap keeps its original fit.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--initial-long-fit", type=Path,
                        help="Saved Lorentz-fit NPZ used only as an explicit numerical initial guess.")
    parser.add_argument("--previous-validation", type=Path,
                        help="Optional earlier supplement's validation.json to merge by case label.")
    args = parser.parse_args(argv)
    origin, destination = args.run_directory.resolve(), args.output.resolve()
    if destination.exists():
        parser.error("Use a new output directory; existing artifacts are preserved.")
    manifest_path = origin / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = [v for v in original["steps"].values() if v.get("status") == "complete"]
    for record in complete:
        path = Path(record["output"])
        if not path.is_file() or sha(path) != record["sha256"]:
            parser.error(f"Original completed artifact is missing or changed: {path}")
    original_config = original["identity"]["config"]
    config = deepcopy(original_config)
    if args.spatial_order < config["numerics"]["spatial_order"]:
        parser.error("This supplement only permits spatial refinement.")
    if args.material_modes not in config["validation"]["shape_check_mode_candidates"]:
        parser.error("Use a pole count declared for shape validation in the original inputs.")
    config["numerics"]["spatial_order"] = args.spatial_order
    config["numerics"]["max_spatial_order"] = max(args.spatial_order, config["numerics"]["max_spatial_order"])
    config["numerics"]["workers"] = args.workers
    config["output"]["directory"] = str(destination)
    initial_guess = None
    if args.initial_long_fit:
        source = args.initial_long_fit.resolve()
        with np.load(source, allow_pickle=False) as fit:
            modes = np.column_stack((fit["strengths_au2"]*AU_ENERGY_EV**2,
                                     fit["omega_modes_au"]*AU_ENERGY_EV,
                                     fit["gamma_modes_au"]*AU_ENERGY_EV))
        if modes.shape != (args.material_modes, 3):
            parser.error("The initial fit pole count must match --material-modes.")
        if "gap_low" in args.cases:
            parser.error("Use a separate invocation for the nominal gap: it keeps its original material fit.")
        config["material"]["refinement"]["initial_modes_eV"] = {"long": modes.tolist()}
        initial_guess = {"path": str(source), "sha256": sha(source),
                         "role": "Initial guess only; the response is refitted to each varied shape"}
    run = ArticleRun(config, destination)
    (destination / "figures").mkdir(exist_ok=True)
    selection = original["state"]["selection"]
    channels = list(dict.fromkeys([selection["best_channel"], selection["runner_up_channel"]]))
    gaps = sorted(set([selection["best_gap_nm"], selection["runner_up_gap_nm"]]))
    master, _ = read_npz(original["state"]["threshold_master"])
    baseline = threshold_subset(master, channels, gaps)
    records = deepcopy(original["state"]["validation"])
    inherited = None
    if args.previous_validation:
        previous_path = args.previous_validation.resolve()
        previous_receipt = json.loads((previous_path.parent / "rerun_receipt.json").read_text(encoding="utf-8"))
        if previous_receipt["original_manifest"]["sha256"] != sha(manifest_path):
            parser.error("The previous supplement belongs to a different original run.")
        expected = previous_receipt["artifacts"].get(str(previous_path))
        if expected is None or sha(previous_path) != expected:
            parser.error("Previous supplement validation hash does not match its receipt.")
        for artifact, digest in previous_receipt["artifacts"].items():
            if not Path(artifact).is_file() or sha(Path(artifact)) != digest:
                parser.error(f"Previous supplement artifact is missing or changed: {artifact}")
        records = json.loads(previous_path.read_text(encoding="utf-8"))
        inherited = {"path": str(previous_path), "sha256": sha(previous_path),
                     "receipt_sha256": sha(previous_path.parent / "rerun_receipt.json")}
    receipt_path = destination / "rerun_receipt.json"
    receipt = {
        "scope": "Supplement for specified sensitivity cases; original full-run status is preserved",
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "original_manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)},
        "original_complete_artifacts_verified": len(complete),
        "script_sha256": sha(Path(__file__)), "cases": args.cases,
        "inherited_validation": inherited,
        "initial_guess": initial_guess,
        "source_changes_since_original": {
            n: {"original": digest, "current": sha(ROOT / n)}
            for n, digest in original["identity"]["source_sha256"].items()
            if sha(ROOT / n) != digest
        },
        "numerical_changes": {"spatial_order": args.spatial_order,
                              "shape_material_modes": args.material_modes},
        "physical_variations": "Exactly the offsets in the original validation configuration",
        "acceptance_limits_relaxed": False,
    }
    write_json(receipt_path, receipt)
    started = time.perf_counter()
    new_records = []
    with material_fit_cache(origin / "material_fit_cache"):
        for label in args.cases:
            cfg = deepcopy(config)
            varied_gaps = gaps
            overrides = {}
            count = original["state"]["material_modes"]
            if label == "gap_low":
                varied_gaps = [g - cfg["validation"]["gap_offset_nm"] for g in gaps]
            else:
                key, sign = ("c_nm", -1) if label == "c_low" else ("a_nm", 1)
                cfg["geometry"][key] *= 1 + sign * cfg["validation"]["shape_relative_offset"]
                count = args.material_modes
                v = cfg["validation"]
                overrides = {"max-modal-normalized-rms": v["shape_check_max_modal_nrms"],
                             "max-modal-relative-error": v["shape_check_max_modal_relative_error"],
                             "spatial-convergence-rtol": v["shape_check_spatial_rtol"]}
            try:
                if label != "gap_low":
                    for gap in varied_gaps:
                        run.material_spectrum("check_shape_" + label, gap, count=count,
                                              config=cfg, overrides=overrides)
                arguments = {**run.threshold_arguments(config=cfg, gaps=varied_gaps,
                                                       channels=channels, count=count), **overrides}
                artifact = run.step("check_" + label, PREFIX + "calculate_threshold_fluence_gap", arguments)
                record = {"label": label, "artifact": str(artifact),
                          **compare_thresholds(baseline, artifact, required_channels=channels),
                          "material_modes_used": count, "spatial_order_used": args.spatial_order,
                          "replacement_receipt": str(receipt_path)}
                if overrides:
                    record["relaxed_numerical_gates"] = overrides
                    record["gate_scope"] = "Unchanged shape-specific limits from original inputs"
            except Exception as exc:
                record = {"label": label, "accepted": False, "error": str(exc),
                          "replacement_receipt": str(receipt_path)}
            new_records.append(record)
            records = [record if r["label"] == label else r for r in records]
            write_json(destination / "validation.json", records)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    ranking = assess_recommendation_ranking(
        master, selection, records, original_config,
        numerical_accepted=original["state"]["numerical_validation_accepted"],
    )
    write_json(destination / "ranking_validation.json", ranking)
    run.plot_validation(records)
    run.manifest["status"] = "complete" if all(r["accepted"] for r in new_records) else "failed"
    run.manifest["scope"] = receipt["scope"]
    run.save()
    for record in complete:
        if sha(Path(record["output"])) != record["sha256"]:
            raise RuntimeError("An original completed artifact changed during the supplement.")
    receipt.update(status=run.manifest["status"], elapsed_s=time.perf_counter() - started,
                   requested_cases_accepted=all(r["accepted"] for r in new_records),
                   combined_ranking_accepted=ranking["accepted"],
                   original_artifacts_unchanged=True,
                   artifacts={str(p): sha(p) for p in destination.rglob("*")
                              if p.is_file() and p != receipt_path})
    write_json(receipt_path, receipt)
    print(f"Sensitivity supplement: {destination}; combined ranking accepted={ranking['accepted']}")
    return 0 if receipt["requested_cases_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
