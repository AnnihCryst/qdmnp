"""Process-based parallelism for independent ODE solves of the article chain.

Only the scheduling changes: each worker builds the same models from the same
arguments and solves the same equations. Workers reuse the material-fit cache
of the parent through ``QDMNP_MATERIAL_FIT_CACHE``, so no fit is repeated.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import tempfile

CACHE_ENV = "QDMNP_MATERIAL_FIT_CACHE"
_WORKER_STACK: ExitStack | None = None


def resolve_workers(requested: int | None, jobs: int) -> int:
    """0 or negative: all logical CPUs except two; never more workers than jobs."""
    if jobs <= 1:
        return 1
    if requested is None:
        requested = 1
    if requested <= 0:
        requested = max(1, (os.cpu_count() or 2) - 2)
    return max(1, min(int(requested), int(jobs)))


def _initialize_worker(cache_directory: str | None) -> None:
    global _WORKER_STACK
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    if cache_directory:
        from qdmnp.observables.article_fit_cache import material_fit_cache

        _WORKER_STACK = ExitStack()
        _WORKER_STACK.enter_context(material_fit_cache(Path(cache_directory)))


@contextmanager
def shared_fit_cache(create_temporary: bool = True):
    """Use one material-fit cache in this process and its workers.

    Active in this process: reuse it. Announced by the environment (e.g. a
    calculator launched as a subprocess by run_article.py): open that directory.
    Otherwise, for parallel runs, a temporary cache lets workers reuse the fits
    of the parent instead of repeating them.
    """
    from qdmnp.rational_fit import HybridQDPlasmonModel
    from qdmnp.observables.article_fit_cache import material_fit_cache

    directory = os.environ.get(CACHE_ENV)
    active = getattr(HybridQDPlasmonModel._fit_rational_alpha, "__name__", "") == "cached"
    if directory and active:
        yield Path(directory)
    elif directory:
        with material_fit_cache(Path(directory)):
            yield Path(directory)
    elif create_temporary:
        with tempfile.TemporaryDirectory(prefix="qdmnp_fit_cache_") as temporary:
            with material_fit_cache(Path(temporary)):
                yield Path(temporary)
    else:
        yield None


def process_pool(workers: int) -> ProcessPoolExecutor:
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    return ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(os.environ.get(CACHE_ENV),),
    )


def fit_material_job(job: tuple) -> tuple:
    """Fill the shared material-fit cache for one (shape, orientation, pole count).

    The fit depends only on the key used by ``material_fit_cache`` (shape,
    orientation, host, pole count, window, refinement, gates), so a placeholder
    QD distance and dipole reproduce the calculators' fits exactly.
    """
    c_nm, a_nm, eps_m, orientation, n_modes, window, refinement = job
    from qdmnp.rational_fit import HybridQDPlasmonModel, make_params_with_overrides

    params = make_params_with_overrides(c_nm=c_nm, a_nm=a_nm, r_nm=c_nm + a_nm + 10.0, eps_m=eps_m,
                                        orientation=orientation)
    model = HybridQDPlasmonModel(
        params, orientation=orientation, n_modes=int(n_modes), fit_refinement=refinement,
        fit_window_eV=tuple(window), max_fit_normalized_rms=None, max_fit_pointwise_relative_error=None,
        radiative_consistency_policy="ignore",
    )
    fit = model.fit
    return (c_nm, a_nm, orientation, int(n_modes), float(fit.normalized_rms_alpha),
            float(fit.normalized_rms_inv_alpha), float(fit.max_normalized_alpha_error))
