// 1d Forman-Ricci curvature: Apple Metal host declaration (metal-cpp).
//
// Authored by Felipe Engelberger.
//
// PRECISION REGIME: FLOAT32. Apple Metal has no general double (float64) support,
// so this backend computes in float (float32) and widens the result back to
// float64 at the boundary. Parity against the float64 CPU/CUDA reference is exact
// only to float32 tolerance: relative error on the order of 1e-6. This mirrors
// the MLX/Metal finding in code/hyphi/backends/metal_mlx.py and the
// frustra-webgpu WGSL f32 findings.
//
// REFERENCE IMPLEMENTATION: this header and its .cpp/.mm host cannot be compiled
// or validated on the authoring machine (no Metal C++ toolchain wired up). They
// are written to be built and validated on Apple Silicon by a Metal engineer; the
// parity harness asserts the ~1e-6 relative tolerance when built.
//
// THE MATH (node weights = 1, the only notion this native path implements):
//   inv_sqrt[k] = 1 / sqrt(we[k])
//   S[v]        = sum of inv_sqrt[k] over all edges k incident to node v
//   curv[k]     = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])

#ifndef HYPHI_NATIVE_METAL_FORMAN_METAL_HPP
#define HYPHI_NATIVE_METAL_FORMAN_METAL_HPP

#include <cstddef>
#include <cstdint>

namespace hyphi_native {

// Compute the per-edge 1d Forman-Ricci curvature on the Apple Metal GPU (float32).
//
// Inputs (structure-of-arrays, contiguous, length n_edges):
//   ei, ej : int32 edge endpoints, node indices in [0, n_nodes).
//   we     : edge weights as double on input; cast to float32 on the device.
// Output (length n_edges):
//   out    : per-edge curvature, widened from the device float32 result to double.
//
// The device buffers are uploaded once (shared storage), both kernels are
// dispatched (S accumulation, then the per-edge curvature), and the float32
// result is widened to float64 into `out`.
//
// Throws std::runtime_error with an actionable message when no Metal device is
// present (MTL::CreateSystemDefaultDevice() returns null) or when any Metal
// resource (library, pipeline, buffer, command buffer) fails to build. The
// dispatcher lets Python catch this and fall back to the CPU path, so this
// function simply raises rather than attempting any fallback itself.
void compute_forman_1d_metal(std::size_t n_nodes,
                             const std::int32_t* ei,
                             const std::int32_t* ej,
                             const double* we,
                             std::size_t n_edges,
                             double* out);

// True iff a default Metal device is present on this machine, i.e.
// MTL::CreateSystemDefaultDevice() returns non-null. Used by the binding's
// has_metal(). When the Metal backend was not compiled in at all, the binding
// reports false without calling this.
bool metal_device_available();

}  // namespace hyphi_native

#endif  // HYPHI_NATIVE_METAL_FORMAN_METAL_HPP
