// 1d Forman-Ricci curvature: Apple Metal compute kernels (MSL).
//
// Authored by Felipe Engelberger.
//
// PRECISION REGIME: FLOAT32. Metal Shading Language has no general double
// (float64) support on Apple GPUs, so this path computes the entire reduction in
// float (float32). Parity against the float64 CPU/CUDA reference is therefore
// exact only to float32 tolerance: relative error on the order of 1e-6, with the
// absolute error scaling with curvature magnitude. This mirrors the MLX/Metal
// finding in code/hyphi/backends/metal_mlx.py and the frustra-webgpu WGSL f32
// findings. When float64 bit-parity is required, use the CPU or CUDA path.
//
// REFERENCE IMPLEMENTATION: this file cannot be compiled or validated on the
// authoring machine (no Metal C++ toolchain wired up). It is written to be built
// and validated on Apple Silicon by a Metal engineer; the parity harness asserts
// the ~1e-6 relative tolerance when built.
//
// THE MATH (node weights = 1, the only notion this native path implements):
//   inv_sqrt[k] = 1 / sqrt(we[k])
//   S[v] = sum of inv_sqrt[k] over all edges k incident to node v
//   curv[k] = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])
//
// DATA LAYOUT: structure-of-arrays, contiguous, uploaded once. ei/ej are int32
// endpoint arrays of length E; we is the float32 edge-weight array (the host
// widens to/from float64 at the boundary); S is a length-N node accumulator.
//
// ATOMIC-FLOAT APPROACH (the load-bearing kernel design decision):
//   k_accumulate_S scatters inv_sqrt onto BOTH endpoints of every edge. Because
//   many edges share a node, the two writes to S[ei[k]] and S[ej[k]] race across
//   threads, so the accumulation must be atomic. Metal's atomic float support is
//   version-dependent:
//
//     - atomic_fetch_add_explicit on `atomic<float>` (device address space) is
//       only guaranteed on Metal 3.0+ (roughly macOS 13 / iOS 16 and the
//       corresponding Apple GPU families). Where it is available it is the
//       cleanest correct choice and is what this kernel uses (k_accumulate_S),
//       guarded by __METAL_VERSION__ >= 300.
//
//     - On older Metal there is no native atomic float add. The portable
//       correct fallback is a compare-and-swap (CAS) loop over `atomic<uint>`
//       that bit-casts float <-> uint (as_type), retrying until the expected
//       value matches. IEEE-754 float addition is not associative, so the order
//       of accumulation can perturb the low bits, but that perturbation is far
//       below the float32 ~1e-6 parity tolerance we already accept; both code
//       paths therefore satisfy the documented regime. The CAS fallback
//       (k_accumulate_S_cas) is provided so the host can pick it when the target
//       device predates Metal 3.0.
//
//   I chose native atomic_float as the primary path (cleanest, correct, fewest
//   instructions) and the atomic_uint CAS bit-cast as the documented fallback,
//   rather than a two-pass sort/segment-reduce approach: the CAS loop keeps the
//   single-dispatch structure-of-arrays design intact and avoids an auxiliary
//   sort buffer, and both approaches land inside the same float32 tolerance.
//
//   The host (forman_metal) is responsible for zero-initializing the S buffer
//   before dispatching k_accumulate_S, and for selecting the native vs CAS
//   variant based on the device's supported Metal version.

#include <metal_stdlib>
using namespace metal;

// Pass 1: scatter-accumulate the per-node inverse-sqrt weight sum S.
// One thread per edge. Each thread adds inv_sqrt = rsqrt(we[k]) onto S at both
// endpoints ei[k] and ej[k] with an atomic add (writes from different edges that
// share a node race, hence the atomic).
//
// Native atomic-float variant (Metal 3.0+).
#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 300)
kernel void k_accumulate_S(
    device const int*    ei        [[buffer(0)]],
    device const int*    ej        [[buffer(1)]],
    device const float*  we        [[buffer(2)]],
    device atomic_float* S         [[buffer(3)]],
    constant uint&       n_edges   [[buffer(4)]],
    uint                 gid       [[thread_position_in_grid]])
{
    if (gid >= n_edges) {
        return;
    }
    // Exact 1 / sqrt(we[k]) to match every other backend (rsqrt is an
    // approximate reciprocal-sqrt and would add error beyond float32 rounding).
    const float inv_sqrt = 1.0f / sqrt(we[gid]);
    const int u = ei[gid];
    const int v = ej[gid];
    atomic_fetch_add_explicit(&S[u], inv_sqrt, memory_order_relaxed);
    atomic_fetch_add_explicit(&S[v], inv_sqrt, memory_order_relaxed);
}
#endif

// Portable fallback: atomic_uint compare-and-swap (CAS) bit-cast float add.
// Correct on older Metal that lacks native atomic float add. Loops a
// compare_exchange on the uint bit pattern of the float until it wins the race.
// The host dispatches this variant when the device predates Metal 3.0; it always
// compiles so the host can choose at runtime.
inline void atomic_add_float_cas(device atomic_uint* cell, float value)
{
    uint expected = atomic_load_explicit(cell, memory_order_relaxed);
    for (;;) {
        const float desired_f = as_type<float>(expected) + value;
        const uint  desired_u = as_type<uint>(desired_f);
        if (atomic_compare_exchange_weak_explicit(
                cell, &expected, desired_u,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
        // On failure `expected` is refreshed with the current value; retry.
    }
}

kernel void k_accumulate_S_cas(
    device const int*    ei        [[buffer(0)]],
    device const int*    ej        [[buffer(1)]],
    device const float*  we        [[buffer(2)]],
    device atomic_uint*  S         [[buffer(3)]],  // float bits, CAS-accumulated
    constant uint&       n_edges   [[buffer(4)]],
    uint                 gid       [[thread_position_in_grid]])
{
    if (gid >= n_edges) {
        return;
    }
    const float inv_sqrt = 1.0f / sqrt(we[gid]);
    const int u = ei[gid];
    const int v = ej[gid];
    atomic_add_float_cas(&S[u], inv_sqrt);
    atomic_add_float_cas(&S[v], inv_sqrt);
}

// Pass 2: per-edge curvature from the completed node-sum S.
// One thread per edge. No atomics: each edge reads S at its two endpoints (which
// are final after pass 1 has finished and the host has inserted a barrier between
// the two dispatches) and writes its own output slot.
//
//   out[k] = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])
//
// S is read as plain float here (the host hands the same backing buffer it
// atomically accumulated into; the bit pattern is a valid float regardless of
// which accumulation variant was used).
kernel void k_edge_curv(
    device const int*   ei        [[buffer(0)]],
    device const int*   ej        [[buffer(1)]],
    device const float* we        [[buffer(2)]],
    device const float* S         [[buffer(3)]],
    device float*       out       [[buffer(4)]],
    constant uint&      n_edges   [[buffer(5)]],
    uint                gid       [[thread_position_in_grid]])
{
    if (gid >= n_edges) {
        return;
    }
    const int u = ei[gid];
    const int v = ej[gid];
    out[gid] = 4.0f - sqrt(we[gid]) * (S[u] + S[v]);
}
