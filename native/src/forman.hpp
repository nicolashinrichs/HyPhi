// HyPhi native curvature core (1d Forman-Ricci) -- shared declarations.
//
// This header declares the memory-safe C++ entry points the `hyphi_native`
// extension module dispatches over: the double-precision CPU reference and the
// (optional, compile-gated) CUDA and Metal accelerator entries. It is the single
// place the frozen native interface is fixed, so every translation unit agrees on
// the signatures. See NATIVE_BACKEND_DESIGN.md for the math, the SoA layout, and
// the parity-before-speed methodology.
//
// THE MATH (1d Forman-Ricci, node weights = 1, the only notion the native path
// implements). Given undirected weighted edges (ei[k], ej[k]) with weight we[k]:
//
//     inv_sqrt[k] = 1 / sqrt(we[k])
//     S[v]        = sum of inv_sqrt[k] over all edges k incident to node v
//     curv[k]     = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])
//
// S is a scatter-add (atomicAdd on the GPU) over BOTH endpoints of every edge.
// This is the validated closed form: it matches
// GraphRicciCurvature.FormanRicci(method="1d") to 1e-13 in float64.
//
// PRECISION REGIME:
//   CPU and CUDA compute in double (float64) -> exact parity with the reference.
//   Metal computes in float (float32) because Metal has no general double support
//   -> parity only to float32 tolerance (relative error ~1e-6). This mirrors the
//   MLX/Metal and frustra-webgpu WGSL f32 findings.
//
// CONVENTIONS (C++ Core Guidelines, mirrored from frustrapy_native): RAII for
// every device resource, no naked malloc/free across scopes, structure-of-arrays
// contiguous data uploaded once. The accelerator entries take plain
// (const T*, int) buffers and an output pointer because the .cu / .mm translation
// units are compiled by nvcc / clang and must not depend on <span> ABI details;
// the host side validates lengths before calling them.
//
// BUILD STATUS: the CUDA (src/cuda/forman_cuda.cu) and Metal
// (src/metal/forman_metal.mm) paths are a reference implementation. They are
// written to mirror the CPU reference term for term, but they have NOT been
// compiled or validated on the current machine (no nvcc, no Metal C++ toolchain
// wired up here). A CUDA/Metal engineer must build and validate them on the target
// accelerator; the parity harness (parity/bench_native.py) runs and asserts the
// tolerances when the extension is built.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hyphi_native {

// Double-precision CPU reference. This is the parity oracle for the native path:
// it reproduces the closed form exactly and every accelerator entry is gated
// against it. SoA inputs (all length n_edges); `out` is caller-allocated, also
// length n_edges, and written in edge order.
//
//   n_nodes : number of graph nodes (S is length n_nodes)
//   ei, ej  : int32 endpoint indices, length n_edges, values in [0, n_nodes)
//   we      : float64 edge weights, length n_edges, strictly positive
//   n_edges : edge count E
//   out     : float64 output buffer, length n_edges (per-edge curvature)
void compute_forman_1d_cpu(int n_nodes, const std::int32_t* ei, const std::int32_t* ej,
                           const double* we, int n_edges, double* out);

#ifdef HYPHI_NATIVE_CUDA
// CUDA double-precision entry (src/cuda/forman_cuda.cu). Uploads the SoA edge
// arrays once, scatter-adds S with atomicAdd, computes curv per edge, copies back.
// Throws std::runtime_error (with an actionable message) if no CUDA device is
// present or a CUDA call fails; the Python wrapper catches and falls back to CPU.
void compute_forman_1d_cuda(int n_nodes, const std::int32_t* ei, const std::int32_t* ej,
                            const double* we, int n_edges, double* out);
#endif

#ifdef HYPHI_NATIVE_METAL
// Metal single-precision entry (src/metal/forman_metal.cpp, metal-cpp). Computes in
// float32 (Metal has no general double support) and writes float64 `out` widened
// from the float32 result, so parity holds only to float32 tolerance (~1e-6
// relative). Throws std::runtime_error if no Metal device is present or a Metal call
// fails; the Python wrapper catches and falls back to CPU.
//
// NOTE: this entry takes std::size_t lengths (matching the metal-cpp definition in
// forman_metal.cpp); the dispatcher widens its int counts to std::size_t. The
// helper metal_device_available() reports a runtime device probe and is used to let
// the dispatcher raise an actionable message before launching.
void compute_forman_1d_metal(std::size_t n_nodes, const std::int32_t* ei,
                             const std::int32_t* ej, const double* we,
                             std::size_t n_edges, double* out);
bool metal_device_available();
#endif

// Compile-time capability flags. has_cuda()/has_metal() report whether the
// corresponding accelerator path was compiled in (HYPHI_NATIVE_CUDA /
// HYPHI_NATIVE_METAL), NOT whether a device is present at runtime; presence is
// discovered when the accelerator entry runs and raises on absence.
bool has_cuda();
bool has_metal();

// Device dispatcher. device is one of {"cpu", "cuda", "metal"}. Returns the
// per-edge curvature as an owning std::vector (length n_edges).
//
// "cpu"   -> compute_forman_1d_cpu (always available).
// "cuda"  -> compute_forman_1d_cuda, behind HYPHI_NATIVE_CUDA; if not compiled in,
//            throws std::runtime_error("... not compiled in ...").
// "metal" -> compute_forman_1d_metal, behind HYPHI_NATIVE_METAL; same contract.
//
// The dispatcher does NOT implement fallback: it raises so Python decides. The
// accelerator entries may also raise if their device is absent at runtime; the
// Python wrapper catches both and falls back to CPU.
std::vector<double> forman_1d_dispatch(int n_nodes, const std::int32_t* ei,
                                       const std::int32_t* ej, const double* we,
                                       int n_edges, const std::string& device);

}  // namespace hyphi_native
