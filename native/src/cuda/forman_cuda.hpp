// forman_cuda.hpp - CUDA host entry for 1d Forman-Ricci edge curvature.
//
// Authored by Felipe Engelberger.
//
// Precision regime: this CUDA path computes in double (float64), giving exact
// parity with the GraphRicciCurvature FormanRicci(method="1d") reference to about
// 1e-13. (The CPU path also computes in double; only the Metal path drops to
// float32 because Metal has no general double support.)
//
// Status: REFERENCE IMPLEMENTATION. This file is correct-by-construction and
// well commented, but it has NOT been compiled or validated on this machine
// (no nvcc, no NVIDIA GPU wired up here). A build-and-validate pass on an NVIDIA
// GPU is required before the CUDA path is trusted in production. The parity
// harness is written to run and assert once the extension is built with CUDA.
//
// This header declares the single host-side entry point. It is C++-includable
// from the nanobind dispatcher (bindings.cpp); the .cu translation unit provides
// the definition and owns all device-side detail (kernels, RAII buffers).

#ifndef HYPHI_NATIVE_CUDA_FORMAN_CUDA_HPP
#define HYPHI_NATIVE_CUDA_FORMAN_CUDA_HPP

#include <cstdint>

namespace hyphi_native {

// Compute 1d Forman-Ricci curvature for every undirected weighted edge on a CUDA
// device, in double precision.
//
// Math (node weights = 1, the only notion the native path implements):
//   inv_sqrt[k] = 1 / sqrt(we[k])
//   S[v]        = sum of inv_sqrt[k] over all edges k incident to node v
//   out[k]      = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])
//
// Parameters (structure-of-arrays, host pointers; uploaded once inside):
//   n_nodes : number of graph nodes; defines the length of the internal S array.
//   ei, ej  : int32 endpoint indices for each of the n_edges edges, length n_edges.
//             Each index must satisfy 0 <= idx < n_nodes.
//   we      : float64 edge weights, length n_edges. Must be strictly positive
//             (the closed form divides by sqrt(we)).
//   n_edges : number of edges (E).
//   out     : float64 output buffer, length n_edges, filled with out[k] above.
//
// Throws std::runtime_error (with an actionable message) if no CUDA device is
// present or any CUDA runtime call fails. The Python wrapper is expected to catch
// that and fall back to the CPU path; this function does not fall back itself.
void compute_forman_1d_cuda(int n_nodes, const std::int32_t* ei,
                            const std::int32_t* ej, const double* we, int n_edges,
                            double* out);

}  // namespace hyphi_native

#endif  // HYPHI_NATIVE_CUDA_FORMAN_CUDA_HPP
