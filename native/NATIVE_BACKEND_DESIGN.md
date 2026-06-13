# HyPhi native backend: design

Status: the CPU reference core, the nanobind bindings, the scikit-build-core / CMake
packaging, and the parity harness are complete and buildable on any machine with a
C++ compiler and Python. The CUDA (`src/cuda/forman_cuda.{cu,hpp}`) and Metal
(`src/metal/forman_metal.{cpp,hpp}`, `src/metal/forman.metal`) paths are a reference
implementation: written to mirror the CPU core term for term, but not yet compiled or
validated on the accelerator hardware (this machine has no nvcc and no Metal C++
toolchain wired up). Parity-before-speed: every step gates on numbers against the
reference, not on speed.

This is the design for a compiled native compute core that plugs in behind HyPhi's
`CurvatureBackend` interface (`code/hyphi/backends/base.py`). The reference path is
the validated NumPy backend (`backends/cpu_numpy.py`); the native core reimplements
the same 1d Forman-Ricci closed form in memory-safe C++ (CPU now, CUDA and Metal as
optional accelerators) and is parity-gated against that reference.

## 1. What the native core reproduces

The geometric core of HyPhi reduces, for the 1-dimensional Forman-Ricci notion, to a
per-edge neighborhood aggregation expressible as a handful of array operations. The
native core implements exactly that one notion (the `augmented` notion stays in
Python, where the sparse triangle/face machinery already lives, and the dispatcher
routes it to NumPy). The closed form is:

```
inv_sqrt[k] = 1 / sqrt(we[k])
S[v]        = sum of inv_sqrt[k] over all edges k incident to node v
curv[k]     = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])
```

`S` is a scatter-add over BOTH endpoints of every edge (an `atomicAdd` on the GPU).
This is the validated closed form: it matches
`GraphRicciCurvature.FormanRicci(method="1d")` to 1e-13 in float64, and it is the same
expression the NumPy, CuPy, and MLX backends already compute. The native core is the
compiled C++/CUDA/Metal equivalent, validated against the same reference.

The derivation (from `backends/base.py`): for an undirected weighted graph with edge
weight `w(e)` on `e = (u, v)` and node weights 1, the combinatorial 1d Forman-Ricci
curvature is

```
F(u, v) = w(e) * ( 1/w(e) + 1/w(e)
                   - sum_{x ~ u, x != v} 1 / sqrt(w(e) * w(u, x))
                   - sum_{y ~ v, y != u} 1 / sqrt(w(e) * w(v, y)) )
```

which, with `S(u) = sum_{x ~ u} 1 / sqrt(w(u, x))`, telescopes to the vectorizable
`F(u, v) = 4 - sqrt(w(e)) * (S(u) + S(v))` above. The `- v` / `- u` exclusions cancel
against the two `1/w(e)` terms, which is why the closed form sums `S` over all incident
edges rather than excluding the edge itself.

## 2. Precision regimes (state in every header and doc)

| Device | Compute dtype | Parity vs the float64 reference |
|--------|---------------|---------------------------------|
| CPU    | float64       | exact (~1e-13) |
| CUDA   | float64       | exact (~1e-13) |
| Metal  | float32       | float32 tolerance only (~1e-6 relative) |

CPU and CUDA compute in double, so they reach exact parity with the reference. Metal
computes in float because Metal Shading Language has no general double support on
Apple GPUs; parity there is exact only to float32 tolerance (relative error on the
order of 1e-6, absolute error scaling with curvature magnitude). This mirrors the
MLX/Metal finding in `code/hyphi/backends/metal_mlx.py` and the frustra-webgpu WGSL
f32 findings. The parity harness asserts `max abs err < 1e-10` for CPU and CUDA and a
looser `< 1e-3` for Metal float32.

## 3. Data layout: structure of arrays (SoA)

All hot data is SoA, contiguous, and uploaded once, so the CPU vectorizes and the GPU
coalesces. One undirected weighted graph is three parallel arrays plus a node count,
the same `GraphArrays` shape the Python backends already use
(`code/hyphi/backends/graph_io.py`):

```
n_nodes : int                      number of graph nodes
ei, ej  : int32[E]                 endpoint indices of each edge, values in [0, n_nodes)
we      : float64[E]               strictly positive edge weights
```

The internal accumulator `S` is `float64[n_nodes]` (CPU/CUDA) or `float32[n_nodes]`
(Metal). The output is `float64[E]` in edge order. There is no array-of-structs and no
pointer arithmetic across a function boundary: the host validates lengths and dtypes at
the nanobind boundary, and the accelerator entries take plain `(const T*, int)` /
`(const T*, std::size_t)` buffers only because the `.cu` / `.cpp` translation units are
compiled by nvcc / clang and must not depend on `<span>` ABI details. Device memory is
held in move-only RAII buffers (`cudaMalloc`/`cudaFree` in a `DeviceBuffer<T>` on CUDA;
metal-cpp `MTL::Buffer` with `NS::SharedPtr`-style release on Metal), so there is no
naked malloc/free and buffers are released even if a later check throws.

## 4. Kernel structure

The computation is two passes over the edge array, both embarrassingly parallel over
edges:

1. **Accumulate S.** For each edge `k`, compute `inv_sqrt[k] = 1/sqrt(we[k])` and add
   it to `S[ei[k]]` and `S[ej[k]]`. This is a histogram / scatter-add: two edges can
   touch the same node concurrently, so on the GPU the two adds are `atomicAdd`. On CPU
   it is a plain sequential accumulation in fixed edge order.
2. **Per-edge curvature.** For each edge `k`, gather `S[ei[k]]` and `S[ej[k]]` and write
   `out[k] = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])`.

CUDA: TPB = 256 threads per block, grid = `(E + TPB - 1) / TPB`, one thread per edge in
both kernels. Double `atomicAdd` requires compute capability >= 6.0 (Pascal and newer),
made explicit by a build-time guard in the `.cu`. Metal: two compute pipelines (the
accumulation kernel, then the curvature kernel) over the same shared buffers, dispatched
with a 1d threadgroup grid; the float32 result is widened to float64 at the boundary.
The two-pass split (rather than fusing) keeps each kernel a simple coalesced sweep and
matches the CPU reference order exactly, which is what the parity gate checks.

## 5. Host-side determinism note

The CPU reference accumulates `S` in a fixed edge order (0 .. E-1) in double precision,
so its result is bit-for-bit deterministic across runs and is the parity oracle. The
CUDA path uses `atomicAdd`, whose floating-point summation order across concurrent
threads is not fixed; in double precision the reordering stays within rounding of the
reference, far below the 1e-10 parity gate, so CUDA is treated as exact. The Metal path
is float32 throughout, so it carries both the reordering noise and the float32
rounding; its gate is the looser 1e-3. None of the paths depend on a random seed (the
kernel is deterministic given the input), so the only nondeterminism is GPU atomic
reordering, which the gates absorb. This is the determinism contract the parity harness
and any downstream reproducibility claim rely on.

## 6. Parity-before-speed methodology

Every step gates on numbers, not on speed. The order is:

1. Build the CPU core and assert it matches a pure-NumPy closed-form reference (the
   same expression, computed independently) to `max abs err < 1e-10` on a panel of
   weighted Watts-Strogatz graphs. The NumPy reference is itself gated bit-for-bit
   against `GraphRicciCurvature` in the main package's backend tests, so this chains the
   native core to the published reference.
2. When built with CUDA, run the same panel on `device="cuda"` and assert the same
   `< 1e-10`. When built with Metal, run on `device="metal"` and assert `< 1e-3`
   (float32). The harness skips a device whose path was not compiled in (`has_cuda()` /
   `has_metal()` are `False`) and prints why.
3. Only after parity holds does any speed work happen; the harness prints a timing
   column, but timings are descriptive, never a gate, and no accelerator timing is
   committed from a machine that lacks the accelerator.

The harness is `parity/bench_native.py`. It is written to RUN and ASSERT when the
extension is importable, and to print a clear "native extension not built; run
`pip install ./native`" message and exit 0 when it is not, so it never fakes numbers.

## 7. Python binding surface (nanobind, zero-copy in)

The compiled module is the self-contained top-level extension `hyphi_native`, built by
this `native/` subproject. nanobind exchanges NumPy arrays zero-copy on input via the
buffer / DLPack protocol; inputs are constrained to contiguous CPU arrays of fixed
dtype:

```cpp
using EdgeIdx = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using EdgeW   = nb::ndarray<const double,  nb::ndim<1>, nb::c_contig, nb::device::cpu>;

forman_1d(int n_nodes, EdgeIdx ei, EdgeIdx ej, EdgeW we, std::string device = "cpu")
    -> float64 ndarray[E];
has_cuda() -> bool;
has_metal() -> bool;
```

The output float64 array owns C++-allocated memory through an `nb::capsule` whose
deleter frees the backing `std::vector` when the NumPy array expires (the nanobind
ownership rule: never return a view of freed storage; always attach an owner). The
binding validates `|ei| == |ej| == |we|` and a non-negative `n_nodes` before any
pointer reaches the core.

The dispatcher (`forman_1d_dispatch` in `core.cpp`) routes `device` to the CPU core or,
behind `#ifdef HYPHI_NATIVE_CUDA` / `#ifdef HYPHI_NATIVE_METAL`, to the accelerator
entry. A device that was not compiled in raises a `std::runtime_error` with an
actionable rebuild message; an accelerator entry may also raise if its hardware is
absent at runtime. The dispatcher does NOT implement fallback: it raises so Python
decides. `hyphi.backends.NativeExtBackend` catches that and falls back to CPU (or, one
level up, to the pure-NumPy backend), so a caller never special-cases device support.

## 8. Build system: scikit-build-core + nanobind, conditional accelerators

`native/` is self-contained so the main `hyphi` package keeps its pure-Python build
untouched. `pip install ./native` produces the extension.

- `native/pyproject.toml`: `build-backend = "scikit_build_core.build"`,
  `requires = ["scikit-build-core>=0.10", "nanobind>=2.0"]`, with the two accelerator
  options exposed as `cmake.define` passthroughs (default OFF).
- `native/CMakeLists.txt`:
  - `find_package(Python ... Development.Module)`, `find_package(nanobind CONFIG)`
    (located via `python -m nanobind --cmake_dir`).
  - `nanobind_add_module(hyphi_native STABLE_ABI NB_STATIC ...)` (stable ABI: one wheel
    per platform across Python minor versions).
  - CUDA off by default: `option(HYPHI_NATIVE_CUDA "..." OFF)`. When on,
    `include(CheckLanguage); check_language(CUDA)`; if nvcc is present,
    `enable_language(CUDA)`, compile `src/cuda/forman_cuda.cu`, define
    `HYPHI_NATIVE_CUDA`; if absent, warn and build CPU-only (graceful, never a hard
    failure) so `has_cuda()` returns `False`.
  - Metal off by default and Apple-only: `option(HYPHI_NATIVE_METAL "..." OFF)`. When on
    and `APPLE`, compile `src/metal/forman_metal.cpp` (metal-cpp, plain C++, no
    Objective-C++ needed), link the Metal / Foundation / QuartzCore frameworks, and
    locate metal-cpp headers via `HYPHI_METAL_CPP_DIR` (or the `METAL_CPP_DIR` env var).
    The MSL shader is embedded as a string literal and compiled at runtime with
    `newLibrary`, so no `.metallib` build step is required; `src/metal/forman.metal` is
    the standalone copy for reference or an optional offline build.
  - A `HYPHI_NATIVE_SANITIZE` option adds `-fsanitize=address,undefined` for the CPU
    parity build.

### Build matrix

| Config | Command | Compiles | has_cuda / has_metal | Validated here |
|--------|---------|----------|----------------------|----------------|
| CPU only (default) | `pip install ./native` | core + bindings | False / False | yes (CPU) |
| + CUDA | `pip install ./native -C cmake.define.HYPHI_NATIVE_CUDA=ON` | + `forman_cuda.cu` | True / False | no (needs nvcc + GPU) |
| + Metal | `... -C cmake.define.HYPHI_NATIVE_METAL=ON` | + `forman_metal.cpp` | False / True | no (needs Apple + metal-cpp) |
| + both | both defines | + both | True / True | no |
| CPU sanitized | `... -C cmake.define.HYPHI_NATIVE_SANITIZE=ON` | core + bindings (ASan/UBSan) | False / False | yes (CPU) |

## 9. Honest accounting: what is NOT yet validated

- The CPU path, the bindings, the packaging, and the parity harness are complete and
  run on any machine with a C++ compiler and Python. The CPU parity assertion
  (`< 1e-10`) is real and runs.
- The CUDA path is a reference implementation. It has NOT been compiled with nvcc or run
  on an NVIDIA GPU on the authoring machine. The `atomicAdd` determinism argument
  (section 5) is reasoned, not measured; a build-and-validate pass on CUDA hardware,
  running `parity/bench_native.py` with `has_cuda() == True`, is required before the
  CUDA result is trusted.
- The Metal path is a reference implementation. It has NOT been compiled with the Metal
  C++ toolchain or run on an Apple GPU on the authoring machine. The float32 tolerance
  (`< 1e-3`) is the expected regime by analogy to the MLX backend, not a measured number
  for this code; a build-and-validate pass on Apple Silicon is required before the Metal
  result is trusted.
- The native path implements the `1d` notion only. `augmented` (AFRC) and any
  Ollivier-Ricci work stay in Python and are out of scope for this core.
- No accelerator timing is committed. The harness prints timings when run, but a timing
  is only meaningful on the hardware that produced it, and none was produced here.

The discipline is the same one HyPhi applies to every number: a claim defaults to
unverified until something is actually run on the relevant hardware, and the docs say so
plainly rather than implying a validation that did not happen.
