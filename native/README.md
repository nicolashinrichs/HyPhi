# hyphi-native

Native compute core for HyPhi's 1d Forman-Ricci curvature kernel. One small,
validated closed form, compiled with nanobind + scikit-build-core + CMake, with an
always-on double-precision CPU path and two optional accelerator paths (CUDA in
float64, Metal in float32). The extension exposes a single compute entry,
`forman_1d`, plus two compile-time capability probes.

This subproject is self-contained: building it does not touch the pure-Python
`hyphi` package. The Python backend `hyphi.backends.NativeExtBackend` (written
separately) imports this extension lazily and falls back to NumPy if it was never
built, so installing `hyphi` without a C++ toolchain still works.

## What it computes

1d Forman-Ricci curvature with node weights set to 1, the only notion the native
path implements. For undirected weighted edges `(ei[k], ej[k])` with weight `we[k]`:

```
inv_sqrt[k] = 1 / sqrt(we[k])
S[v]        = sum of inv_sqrt[k] over all edges k incident to node v   (scatter-add)
curv[k]     = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])
```

This is the validated closed form: it matches
`GraphRicciCurvature.FormanRicci(method="1d")` to 1e-13 in float64. The same form
runs in `hyphi/backends/cpu_numpy.py` (NumPy), `cuda_cupy.py` (CuPy), and
`metal_mlx.py` (MLX); this extension is the compiled C++/CUDA/Metal equivalent.

## Public interface

```python
import numpy as np
import hyphi_native

hyphi_native.has_cuda()   # bool: was the CUDA path compiled in?
hyphi_native.has_metal()  # bool: was the Metal path compiled in?

n_nodes = 5
ei = np.array([0, 1, 2, 3], dtype=np.int32)
ej = np.array([1, 2, 3, 4], dtype=np.int32)
we = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)

curv = hyphi_native.forman_1d(n_nodes, ei, ej, we, device="cpu")  # float64[E]
```

`device` is one of `"cpu"`, `"cuda"`, `"metal"`. Inputs are contiguous CPU arrays:
`ei`, `ej` are int32 of length E, `we` is float64 of length E, and the three lengths
must match. The output is a float64 array of length E in edge order. A device that
was not compiled in, or that has no hardware present at runtime, raises a
`RuntimeError`; the Python wrapper catches that and falls back to the CPU path.

## Precision regimes (state this everywhere)

| Device | Compute dtype | Parity vs the float64 reference |
|--------|---------------|---------------------------------|
| cpu    | float64       | exact (~1e-13) |
| cuda   | float64       | exact (~1e-13); double `atomicAdd` reordering stays far below the 1e-10 gate |
| metal  | float32       | float32 tolerance only (~1e-6 relative); Metal has no general double support |

The Metal float32 regime mirrors the MLX/Metal finding in
`code/hyphi/backends/metal_mlx.py` and the frustra-webgpu WGSL f32 findings. Use the
CPU or CUDA path when float64 bit-parity matters; use Metal when float32 throughput on
Apple Silicon matters more than reproducibility.

## How to build

CPU-only (always works, no nvcc or Metal toolchain needed):

```
pip install ./native
```

With the CUDA path (requires the CUDA Toolkit / nvcc on the build machine):

```
pip install ./native -C cmake.define.HYPHI_NATIVE_CUDA=ON
```

With the Metal path (requires Apple Silicon and a metal-cpp checkout; point
`HYPHI_METAL_CPP_DIR` or the `METAL_CPP_DIR` env var at it):

```
METAL_CPP_DIR=/path/to/metal-cpp pip install ./native \
  -C cmake.define.HYPHI_NATIVE_METAL=ON \
  -C cmake.define.HYPHI_METAL_CPP_DIR=/path/to/metal-cpp
```

Both accelerator options are independent and can be combined. If `HYPHI_NATIVE_CUDA=ON`
is requested but nvcc is absent, the build emits a warning and proceeds CPU-only
(graceful fallback, never a hard failure); `has_cuda()` then reports `False`. The same
graceful behavior applies to Metal off Apple hardware.

An ASan/UBSan build for the CPU parity gate:

```
pip install ./native -C cmake.define.HYPHI_NATIVE_SANITIZE=ON
```

## Reference-implementation status (honest accounting)

The CPU path (`src/core.cpp`), the nanobind bindings (`src/bindings.cpp`), the CMake /
scikit-build packaging, and the parity harness are complete and buildable on any
machine with a C++ compiler and Python.

The CUDA path (`src/cuda/forman_cuda.{cu,hpp}`) and the Metal path
(`src/metal/forman_metal.{cpp,hpp}` plus the standalone `src/metal/forman.metal`
shader) are a **reference implementation**: they are written to mirror the CPU
reference term for term, but they have NOT been compiled or validated on the machine
they were authored on (no nvcc, no Metal C++ toolchain wired up there). A CUDA or
Metal engineer must build and validate them on the target accelerator. The parity
harness (`parity/bench_native.py`) is written to RUN and ASSERT the tolerances when
the extension is built; it never fakes numbers. Until that build-and-validate pass is
done on real hardware, treat the accelerator results as unverified.

## Files

```
native/
  CMakeLists.txt              project; nanobind; optional CUDA / Metal options
  pyproject.toml              scikit-build-core + nanobind build backend
  README.md                   this file
  NATIVE_BACKEND_DESIGN.md    the design doc (math, SoA, kernels, parity, build matrix)
  src/
    forman.hpp                shared declarations (CPU + guarded CUDA/Metal externs)
    core.cpp                  compute_forman_1d_cpu + the device dispatcher
    bindings.cpp              nanobind module hyphi_native (has_cuda/has_metal/forman_1d)
    cuda/
      forman_cuda.hpp         CUDA host entry declaration
      forman_cuda.cu          CUDA kernels (reference implementation, float64)
    metal/
      forman_metal.hpp        Metal host entry declaration
      forman_metal.cpp        metal-cpp host (reference implementation, float32)
      forman.metal            standalone MSL shader (reference / optional .metallib)
  parity/
    bench_native.py           runnable parity + timing harness (asserts tolerances)
```

## How the Python wrapper uses this

`hyphi.backends.NativeExtBackend` (in the main package, not here) imports
`hyphi_native` lazily inside its compute method. It calls `has_cuda()` / `has_metal()`
to learn which device paths were compiled in, calls `forman_1d(...)` with the requested
device, and catches the `RuntimeError` a missing device raises to fall back to the CPU
path (or, one level up, to the pure-NumPy backend) so a caller never has to special-case
device support. That backend reports `is_available()` by trying the import, so a machine
without the extension simply does not offer it.
