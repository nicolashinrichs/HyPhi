"""
Scaling helpers for large curvature runs: thread budget, parallel map, HDF5 store.

When a HyPhi study grows to many dyads x trials x sliding windows, two costs
dominate, and neither is the curvature arithmetic (which the vectorized backends
already make cheap):

1. Thread oversubscription. Each worker process inherits a BLAS/OpenMP pool, so
   ``P`` processes each spawning ``C`` threads contend for ``P * C`` threads on
   ``C`` cores. :func:`limit_blas_threads` pins the math libraries to one thread
   per process, and :func:`resolve_concurrency` allocates a shared ``(outer,
   inner)`` budget so nested pools cannot fork-bomb.

2. Filesystem metadata. Writing one small ``.npy`` per window produces millions
   of inodes; on a cluster filesystem (Lustre, GPFS, NFS) the create/stat/quota
   path, not the bytes, is the bottleneck. :class:`CurvatureStore` writes one
   compressed HDF5 file per shard with a one-writer-per-file discipline, then
   merges shards by metadata copy, collapsing the inode count.

These patterns are adapted from the FrustraPy HPC tooling, validated there at
cluster scale, and re-expressed here for the curvature-series workload.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings

import numpy as np

__all__ = [
    "CurvatureStore",
    "limit_blas_threads",
    "map_curvature_series",
    "resolve_concurrency",
]

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


# Backends holding a per-process GPU device context that does not survive fork/forkserver, so they
# must not be run inside the multiprocessing pool (see map_curvature_series).
_GPU_BACKENDS = frozenset({"mlx", "cupy", "native"})


def limit_blas_threads(n: int = 1) -> None:
    """
    Pin BLAS/OpenMP thread pools to ``n`` threads per process.

    Call this once, before importing or using NumPy in worker processes, so that
    process-level parallelism over a graph series does not collide with
    thread-level parallelism inside each math call. Uses ``setdefault`` so an
    explicit user override is preserved.

    Parameters
    ----------
    n : int
        Threads per process (default 1, the safe choice under multiprocessing).

    """
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n))


def resolve_concurrency(n_procs: int, n_items: int, cores: int | None = None) -> tuple[int, int]:
    """
    Allocate a shared ``(outer, inner)`` concurrency budget.

    Guarantees ``outer * inner <= cores`` so that an outer process pool and an
    inner thread pool cannot oversubscribe the machine.

    Parameters
    ----------
    n_procs : int
        Requested outer (process) parallelism.
    n_items : int
        Number of work items; the outer pool is never larger than this.
    cores : int, optional
        Core budget (default ``os.cpu_count()``).

    Returns
    -------
    tuple of int
        ``(outer, inner)`` with ``outer >= 1``, ``inner >= 1``, and
        ``outer * inner <= cores``.

    """
    if cores is None or cores < 1:
        cores = os.cpu_count() or 1
    n_procs = n_procs if n_procs is not None else 1
    n_items = n_items if n_items is not None else 1
    outer = max(1, min(max(1, n_procs), max(1, n_items), cores))
    inner = max(1, cores // outer)
    return outer, inner


def _safe_start_method() -> str:
    """
    Pick a multiprocessing start method that does not fork a GPU/BLAS process.

    ``fork`` is unsafe after a Metal/MLX context or a threaded BLAS pool has been
    initialized (it can deadlock, notably on macOS). Prefer ``forkserver``, then
    ``spawn``, then ``fork``.
    """
    available = mp.get_all_start_methods()
    for method in ("forkserver", "spawn", "fork"):
        if method in available:
            return method
    return available[0]


def _series_worker(args):
    from . import forman_curvature  # noqa: PLC0415

    warnings.filterwarnings("ignore")
    graph, method, backend, weight = args
    return forman_curvature(graph, method=method, backend=backend, weight=weight)


def map_curvature_series(
    graphs,
    method: str = "1d",
    backend: str | None = None,
    weight: str = "weight",
    n_procs: int = 1,
):
    """
    Compute per-edge curvature over a series of graphs, optionally in parallel.

    For the vectorized array backends the per-graph kernel is so cheap that
    single-process execution is usually fastest (process startup and pickling
    dominate); multiprocessing pays off mainly for the un-vectorized reference
    backend or very large per-graph work. This helper makes both available and
    pins BLAS threads in each worker.

    Parameters
    ----------
    graphs : sequence of networkx.Graph
        The graph series (e.g. sliding windows).
    method : {"1d", "augmented"}
        Curvature notion.
    backend : str or None
        Backend selector (see :func:`hyphi.backends.get_backend`).
    weight : str
        Edge weight attribute.
    n_procs : int
        Outer process parallelism; ``1`` runs in-process.

    Returns
    -------
    list of numpy.ndarray
        Per-graph per-edge curvature arrays, in input order.

    """
    from . import forman_curvature, get_backend  # noqa: PLC0415

    graphs = list(graphs)
    # Resolve the backend in the PARENT. A GPU backend (Metal / CUDA) holds a per-process device
    # context that does not survive fork/forkserver: a worker dies while re-initialising it and
    # pool.map() then blocks forever (observed ~35% of the time with Metal). GPU backends already
    # parallelise within each graph, so the series runs serially for them; only the fork-safe CPU
    # backends use the process pool.
    is_gpu = get_backend(backend or "auto").name in _GPU_BACKENDS
    if n_procs <= 1 or len(graphs) <= 1 or is_gpu:
        return [forman_curvature(g, method=method, backend=backend, weight=weight) for g in graphs]

    outer, _ = resolve_concurrency(n_procs, len(graphs))
    limit_blas_threads(1)
    ctx = mp.get_context(_safe_start_method())
    with ctx.Pool(outer, initializer=limit_blas_threads) as pool:
        return pool.map(_series_worker, [(g, method, backend, weight) for g in graphs])


class CurvatureStore:
    """
    HDF5-backed store for curvature-series results at cluster scale.

    Writes one compressed HDF5 dataset per result group instead of one small
    file per window, collapsing inode pressure on shared filesystems. Each
    parallel writer should own its own shard file (``one writer per file``);
    shards are then merged into a single file by metadata copy with
    :meth:`merge_shards` (no recompression).

    Parameters
    ----------
    path : str
        Output HDF5 path.
    mode : str
        h5py file mode (default ``"a"``).

    Notes
    -----
    ``h5py`` is an optional dependency; install it with ``pip install
    'hyphi[hpc]'``. Datasets use gzip compression with the shuffle filter, which
    helps near-equal float64 curvature values.

    """

    def __init__(self, path: str, mode: str = "a") -> None:
        """Open (or create) the HDF5 store at ``path`` in h5py file ``mode``."""
        self._h5py = self._require_h5py()
        self.path = path
        self._file = self._h5py.File(path, mode)

    @staticmethod
    def _require_h5py():
        try:
            import h5py  # noqa: PLC0415  # ty:ignore[unresolved-import]
        except ImportError as exc:
            raise ImportError("CurvatureStore needs h5py. Install it with: pip install 'hyphi[hpc]'") from exc
        return h5py

    def save(self, key: str, array: np.ndarray, attrs: dict | None = None) -> None:
        """Write ``array`` under ``key`` with gzip+shuffle, plus optional attrs."""
        arr = np.ascontiguousarray(array)
        if key in self._file:
            del self._file[key]
        dset = self._file.create_dataset(
            key,
            data=arr,
            chunks=True,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        for k, v in (attrs or {}).items():
            dset.attrs[k] = v

    def load(self, key: str) -> np.ndarray:
        """Read the dataset stored at ``key`` as a NumPy array."""
        return self._file[key][()]

    def keys(self) -> list[str]:
        """Return the top-level dataset/group keys in the store."""
        return list(self._file.keys())

    def close(self) -> None:
        """Flush and close the underlying HDF5 file."""
        self._file.close()

    def __enter__(self) -> CurvatureStore:
        """Enter the context manager, returning the open store."""
        return self

    def __exit__(self, *exc) -> None:
        """Close the underlying HDF5 file on context exit."""
        self.close()

    @classmethod
    def merge_shards(cls, out_path: str, shard_paths: list[str]) -> None:
        """
        Merge per-writer shard files into one store by metadata copy.

        Each shard's top-level objects are copied into ``out_path`` under a
        per-shard group, avoiding recompression and concurrent-writer hazards.

        Parameters
        ----------
        out_path : str
            Destination HDF5 path.
        shard_paths : list of str
            Shard files written by independent workers (one writer each).

        """
        h5py = cls._require_h5py()
        with h5py.File(out_path, "w") as dst:
            for i, shard in enumerate(shard_paths):
                with h5py.File(shard, "r") as src:
                    grp = dst.create_group(f"shard_{i:04d}")
                    for key in src:
                        src.copy(key, grp)
