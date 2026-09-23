"""Recheck tip and side thresholds with N=13, nmax=160 and tighter ODE steps."""
from qdmnp.pipeline import Tee, write_json, material_fit_cache, sha

from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys
import time

from qdmnp.observables.calculate_threshold_fluence_gap import main


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    origin = directory.parent
    m = json.loads((origin/"manifest.json").read_text(encoding="utf-8"))
    record = next(v for k,v in m["steps"].items() if k.startswith("check_higher_N:")
                  and v["command"][2].endswith("calculate_threshold_fluence_gap"))
    arguments = list(record["command"][3:])
    def replace(flag, values):
        start = arguments.index(flag)+1
        end = start
        while end < len(arguments) and not arguments[end].startswith("--"):
            end += 1
        arguments[start:end] = list(map(str, values))
    changes = {"--channels": ["axis_long", "side_long"], "--spatial-order-max": [160],
               "--points-per-fastest-cycle": [16], "--rtol": [1e-8], "--atol": [1e-10],
               "--workers": [1], "--output": [directory/"thresholds_refined.npz"]}
    for flag, values in changes.items():
        replace(flag, values)
    receipt = {"original_manifest_sha256": sha(origin/"manifest.json"),
               "original_command": record["command"], "arguments": arguments,
               "script_sha256": sha(Path(__file__)), "status": "running"}
    write_json(directory/"threshold_receipt.json", receipt)
    start = time.perf_counter()
    try:
        with (directory/"threshold_calculation.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                with material_fit_cache(origin/"material_fit_cache"):
                    output = main(arguments)
        receipt.update(status="complete", output=str(output), sha256=sha(output))
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        receipt["elapsed_s"] = time.perf_counter()-start
        write_json(directory/"threshold_receipt.json", receipt)
