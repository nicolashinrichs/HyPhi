# Accelerating discrete-curvature computation in HyPhi

This document describes the `hyphi.backends` accelerator layer: why it exists, how
it is structured, the measured evidence that it is correct and fast, and how to
reproduce every number here. It also covers the native CUDA/Metal subproject and
the IO/HPC helpers for large runs.

All timings and parity figures in this document are measured, not asserted, and
were produced by `experiments/scripts/bench_curvature_backends.py`. Re-running
that script regenerates the tables.

## 1. The bottleneck this addresses

The geometric core of HyPhi computes Forman-Ricci curvature on a time series of
inter-brain graphs. The shipped path,
`hyphi.modeling.graph_curvatures.compute_frc_vec`, is a sequential loop that calls
`GraphRicciCurvature.FormanRicci` once per graph. Profiling that call on a
1000-node, 25000-edge graph (the NeurReps replication shape) shows the cost is
not the arithmetic but the per-edge NetworkX adjacency-dictionary traversal:
about 13.75 million Python calls per graph, dominated by `self.G[u][v][weight]`
lookups inside the per-edge neighbor sum, called roughly 2.6 million times per
graph. The full 200-replication NeurReps sweep therefore costs about 2.7 hours,
single threaded, with thirteen of fourteen cores idle.

The 1d Forman curvature, however, has a closed form that is a handful of array
operations, so the bottleneck is removed by vectorization, and the same
vectorized form runs on a GPU.

## 2. The mathematics the backends compute

For an undirected weighted graph with edge weight `w(e)` on edge `e = (u, v)` and
node weights set to 1 (the `GraphRicciCurvature` default and the universal HyPhi
convention), the combinatorial 1d Forman-Ricci curvature is

```
F(u, v) = w(e) * ( 1/w(e) + 1/w(e)
                   - sum_{x ~ u, x != v} 1 / sqrt(w(e) * w(u, x))
                   - sum_{y ~ v, y != u} 1 / sqrt(w(e) * w(v, y)) )
```

Defining the per-node quantity `S(u) = sum_{x ~ u} 1 / sqrt(w(u, x))`, this
telescopes to the vectorizable closed form

```
F(u, v) = 4 - sqrt(w(e)) * ( S(u) + S(v) )                              (1d)
```

which every backend computes: one reciprocal-sqrt per edge, one scatter-add to
accumulate `S` per node, one gather, one fused multiply-add. The augmented notion
(AFRC) additionally needs the per-edge triangle count and a face-restricted
weight sum; the NumPy backend computes both with two sparse matrix products
(`A @ A` for triangle counts, `M @ A` for the face-restricted weight sums with
`M[u, x] = 1 / sqrt(w(u, x))`), giving

```
F_aug(u, v) = |face| * w(e)^2 + 2 - sqrt(w(e)) * |T_u + T_v|            (augmented)
```

These identities are validated bit-for-bit against the reference loop in
`code/tests/test_backends.py` and in the benchmark below.

## 3. Architecture

```
hyphi.backends
  base.CurvatureBackend        abstract interface (name, device_kind, compute_dtype,
                               is_available, forman_curvature) + the math
  graph_io.GraphArrays         structure-of-arrays boundary (ei, ej, we, node_order)
  cpu_numpy.NumpyBackend       vectorized NumPy, float64, 1d + augmented   [reference-equivalent]
  cuda_cupy.CupyBackend        CuPy on NVIDIA CUDA, float64, 1d
  metal_mlx.MlxBackend         MLX on Apple Metal, float32, 1d
  native_ext.NativeExtBackend  optional compiled hyphi_native (C++/CUDA/Metal)
  reference_networkx.NetworkxBackend  GraphRicciCurvature, the parity oracle and safe default
  capabilities.detect / recommend_backend / install_hint   per-machine selection
  hpc.limit_blas_threads / resolve_concurrency / map_curvature_series / CurvatureStore
```

One vectorized kernel, several devices: NumPy, CuPy and MLX run the same closed
form with only the array module swapped. The dispatcher (`get_backend`,
`forman_curvature`) selects a backend by name, by the `HYPHI_BACKEND` environment
variable, or by `backend="auto"` (the fastest accelerator available on the
machine). A backend that does not implement a requested method (the GPU backends
implement 1d only) transparently falls back to the NumPy backend, so callers
never special-case device support. The CPU NumPy backend is the safe default; an
accelerator is opt-in.

## 4. Parity (correctness gated before speed)

Every backend is validated against the `GraphRicciCurvature` reference. The CPU
float64 backends match to machine precision; the Metal/MLX backend computes in
float32 (the Metal GPU has no general float64 support, exactly as MLX and the
WebGPU/WGSL paths report) and matches to float32 tolerance.

Measured on macOS 15.5, Apple Silicon (arm64), 14 logical cores, Python 3.13.5,
NumPy 1.26.4, NetworkX 3.6.1, MLX 0.31.2 (CUDA absent on this machine):

| Graph | nodes | edges | backend | dtype | max abs err | max rel err | Pearson r |
|---|---|---|---|---|---|---|---|
| WS-small | 200 | 600 | numpy | float64 | 8.88e-15 | 8.28e-16 | 1.000000 |
| WS-small | 200 | 600 | mlx | float32 | 2.96e-06 | 3.05e-07 | 1.000000 |
| WS-med | 1000 | 5000 | numpy | float64 | 1.42e-14 | 6.76e-16 | 1.000000 |
| WS-med | 1000 | 5000 | mlx | float32 | 6.01e-06 | 3.12e-07 | 1.000000 |
| WS-large | 2000 | 12000 | numpy | float64 | 2.13e-14 | 7.27e-16 | 1.000000 |
| WS-large | 2000 | 12000 | mlx | float32 | 8.22e-06 | 3.05e-07 | 1.000000 |
| WS-dense | 1000 | 25000 | numpy | float64 | 1.14e-13 | 7.79e-16 | 1.000000 |
| WS-dense | 1000 | 25000 | mlx | float32 | 4.84e-05 | 4.62e-07 | 1.000000 |

The NumPy backend is the reference to floating-point round-off; the Metal backend
agrees to float32 (relative error of order 1e-6, with absolute error scaling with
curvature magnitude). Pearson r is 1.000000 throughout.

### 4.1 Self-loops: a deliberate divergence from the legacy path

The parity tables above use Watts-Strogatz graphs, which have no self-loops. The
shipped Kuramoto connectome graphs, however, carry one self-loop per node (the PLV
diagonal is 1.0, a self-correlation artifact). The backends exclude self-loops by
construction: a self-loop is not a 1-simplex in the Forman sense, and a PLV
self-loop is not an inter-node edge. The legacy `compute_frc`, by contrast,
iterates self-loop edges too. So on a self-loop graph the two paths differ, and
the difference is the self-loop policy, not a kernel error.

Measured on a shipped connectome window (152 nodes):

| Quantity | legacy compute_frc | hyphi.backends | note |
|---|---|---|---|
| edges with a curvature value | 11628 | 11476 | 152 self-loops excluded |
| mean abs FRC | 375.12 | 371.84 | self-loop term removed from S(u) and the 152 self-loop values |
| backend vs legacy on the self-loop-removed graph | - | - | max abs 1.1e-12 (the kernel is exact once the graphs match) |

The backends are the more defensible choice scientifically (self-loops distort the
curvature spectrum and the downstream entropy), but because this changes published
numbers on self-loop data it is called out here and is a question worth a decision:
should the pipeline remove PLV self-loops everywhere (consistent with this layer)
or keep the legacy behavior? `forman_curvature(..., annotate=True)` returns a copy
with self-loops removed and a curvature on every remaining edge; the per-edge array
is aligned to edges through the SoA index, not edge-iteration order, so relabeled
or string-labeled nodes map correctly.

## 5. Throughput (apples-to-apples)

### 5.1 Single graph, median wall time

| Graph | nodes | edges | numpy (CPU f64) | networkx (ref) | mlx (Metal f32) |
|---|---|---|---|---|---|
| WS-small | 200 | 600 | 0.01 ms | 3.50 ms | 0.23 ms |
| WS-med | 1000 | 5000 | 0.03 ms | 38.49 ms | 0.61 ms |
| WS-large | 2000 | 12000 | 0.08 ms | 159.08 ms | 0.70 ms |
| WS-dense | 1000 | 25000 | 1.31 ms | 745.68 ms | 0.56 ms |

On the dense 1000-node graph (the NeurReps replication shape) the vectorized
NumPy backend is about 570x faster than the reference and the Metal backend about
1330x faster. Note the crossover: NumPy beats Metal on small and sparse graphs
because GPU launch and host-to-device transfer overhead exceed the kernel cost
there, while Metal wins once per-graph work is large and dense. Use NumPy for
small or sparse graphs and Metal for large dense graphs; `backend="auto"` is a
reasonable default but is not always the fastest at small sizes.

### 5.2 Series throughput (40 graphs of 1000 nodes, the pipeline shape)

| Path | total time | per graph | speedup vs reference |
|---|---|---|---|
| NetworkX reference (sequential) | 2.19 s | 54.8 ms | 1.0x |
| NumPy vectorized (sequential) | 0.050 s | 1.24 ms | 44x |
| MLX Metal GPU (sequential) | 0.067 s | 1.67 ms | 33x |
| NumPy vectorized + 14-core multiprocessing | 0.222 s | 5.56 ms | 10x |

Two findings worth stating plainly. First, vectorization is the dominant win:
44x on the series, up to 570x on a single dense graph, at exact float64. Second,
multiprocessing helps the slow reference path but is counterproductive once the
kernel is vectorized: the 14-core multiprocessing run is slower than
single-threaded vectorized NumPy because process startup and pickling now exceed
the (tiny) kernel cost. The recommended order is therefore: vectorize first; use
the GPU for large dense graphs; reach for multiprocessing only for the
un-vectorized reference path or for very large per-graph work.

## 6. Precision regimes

| Backend | Device | dtype | Parity vs reference |
|---|---|---|---|
| networkx | CPU | float64 | exact (the oracle) |
| numpy | CPU | float64 | machine precision (<= 1e-13 abs) |
| cupy | NVIDIA CUDA | float64 | machine precision (expected; validate on a CUDA host) |
| mlx | Apple Metal | float32 | ~1e-6 relative |
| native (cpu) | CPU C++ | float64 | machine precision (expected; build required) |
| native (cuda) | CUDA C++ | float64 | machine precision (expected; build required) |
| native (metal) | Metal C++ | float32 | ~1e-6 relative (expected; build required) |

Choose a float64 backend (numpy, cupy, native cpu/cuda) when bit-level
reproducibility matters for a published number; choose a Metal backend when
throughput on large graph series matters more than the last few significant
digits. The entropy and quantile summaries HyPhi computes downstream are
insensitive to float32 curvature round-off.

## 7. Per-platform install (the auto-selection)

`pip install 'hyphi[accel]'` resolves the correct accelerator for the machine
through PEP 508 environment markers: MLX is installed only on Apple Silicon, and
`cupy-cuda12x` only on x86_64 Linux or Windows. There is nothing to choose by
hand.

```
pip install 'hyphi[accel]'   # auto-selects: MLX on Apple Silicon, CuPy on x86_64 Linux/Windows
pip install 'hyphi[metal]'   # Apple Metal only
pip install 'hyphi[cuda]'    # NVIDIA CUDA 12 only (use cupy-cuda11x for a CUDA 11 toolkit)
pip install 'hyphi[hpc]'     # h5py for the HDF5 curvature store
python -m hyphi.backends      # report this machine's capabilities and the exact install hint
python -m hyphi.backends --bench   # also run a quick parity and timing check
```

The CUDA wheel is pinned to the CUDA 12 runtime (`cupy-cuda12x`); a CUDA 11 host
needs `cupy-cuda11x` instead, which the marker cannot detect (it sees the
platform, not the toolkit version). This is the one case that is not fully
automatic, and it is documented rather than guessed.

## 8. IO and HPC for large runs (`hyphi.backends.hpc`)

When a study grows to many dyads by trials by windows, the cost moves from the
curvature arithmetic (now cheap) to thread oversubscription and filesystem
metadata. The helpers, adapted from patterns validated at cluster scale in the
FrustraPy HPC tooling, address both:

- `limit_blas_threads()` pins BLAS/OpenMP to one thread per process, and
  `resolve_concurrency(n_procs, n_items)` returns a shared `(outer, inner)` budget
  with `outer * inner <= cores`, so a process pool and an inner thread pool cannot
  oversubscribe the machine.
- `map_curvature_series(graphs, backend=..., n_procs=...)` computes a series with
  the chosen backend, pinning threads in each worker. As the throughput table
  shows, single-process vectorized execution is usually fastest; multiprocessing
  is for the reference path or very large per-graph work. The **GPU backends
  (`mlx`, `cupy`, `native`) always run the series single-process**, even with
  `n_procs > 1`: a Metal/CUDA device context does not survive `fork`/`forkserver`
  (a worker deadlocks re-initialising it), and the GPU already parallelises within
  each graph. Only the fork-safe CPU backends use the process pool.
- `CurvatureStore` writes results into a single compressed HDF5 file (gzip plus
  the shuffle filter, one writer per shard) instead of one small file per window,
  collapsing the inode pressure that dominates on Lustre, GPFS and NFS.
  `CurvatureStore.merge_shards` merges per-writer shard files by metadata copy,
  the standard one-writer-per-file then merge discipline for SLURM array jobs.

## 9. The native subproject (`native/`)

`native/` is a hand-written-kernel implementation built with nanobind and
scikit-build-core: a C++ CPU reference (`core.cpp`, the float64 parity oracle), a
CUDA `.cu` kernel (`src/cuda/forman_cuda.cu`, float64), and a Metal compute shader
plus metal-cpp host (`src/metal/`, float32). It exposes the same `forman_1d`,
`has_cuda` and `has_metal` symbols the Python `NativeExtBackend` wraps. It is
optional: a plain `pip install hyphi` never needs a compiler, and the pure-Python
array backends cover the same 1d kernel.

```
pip install ./native                                       # CPU C++ reference only
pip install ./native -C cmake.define.HYPHI_NATIVE_CUDA=ON  # + CUDA (needs nvcc, Pascal+ GPU)
pip install ./native -C cmake.define.HYPHI_NATIVE_METAL=ON # + Metal (Apple Silicon, Xcode)
```

Honest accounting: the native CUDA and Metal kernels are a reference
implementation. They were authored against the validated closed form and the
frustrapy native idioms (RAII device buffers, scatter-add accumulation,
parity-before-speed), but they cannot be compiled or validated on the
development machine used here (no nvcc, no Metal C++ toolchain). The parity
harness `native/parity/bench_native.py` is written to build the same graphs,
compute `forman_1d` on every compiled device, and assert max abs error below
1e-10 for the float64 paths and below 1e-3 for the Metal float32 path; it must be
run on the target accelerator to clear those assertions. Until then, the
validated, shipped acceleration is the pure-Python array layer (NumPy exact,
MLX/Metal to float32), and the native path is documented as build-and-validate
required.

## 10. Reproduce

```
# capabilities and a quick parity/timing check on this machine
python -m hyphi.backends --bench

# the full benchmark and parity tables in this document
python experiments/scripts/bench_curvature_backends.py

# the parity test suite (float64 exact, float32 tolerance, fallback, known-answer)
pytest code/tests/test_backends.py -v

# native parity (only on a built CUDA/Metal host)
python native/parity/bench_native.py
```
