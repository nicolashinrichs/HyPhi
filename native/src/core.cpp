// HyPhi native curvature core (1d Forman-Ricci) -- CPU reference and dispatcher.
//
// compute_forman_1d_cpu is the double-precision parity oracle for the native path.
// It reproduces the validated closed form (header forman.hpp) exactly:
//
//     inv_sqrt[k] = 1 / sqrt(we[k])
//     S[v]        = sum of inv_sqrt[k] over edges k incident to node v
//     curv[k]     = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])
//
// matching GraphRicciCurvature.FormanRicci(method="1d") to 1e-13 in float64.
//
// PRECISION REGIME: CPU computes in double (float64) -> exact parity with the
// reference. The CUDA path is also float64 (exact); the Metal path is float32
// (~1e-6 relative). See NATIVE_BACKEND_DESIGN.md.
//
// DETERMINISM NOTE: the scatter-add over S is computed in a fixed edge order
// (0..n_edges-1) in double precision, so the CPU result is bit-for-bit
// deterministic across runs. The CUDA path uses atomicAdd, whose floating-point
// summation order is not fixed; double-precision atomicAdd keeps the result within
// rounding of this reference (the parity gate is 1e-10, far above that noise).
//
// CONVENTIONS (C++ Core Guidelines): no owning raw pointers held across scopes,
// RAII (std::vector) for the S accumulator, bounds-checkable indexing. The raw
// (const T*, int) signatures exist only to match the accelerator entries compiled
// by nvcc / clang; the nanobind binding layer validates lengths and dtypes before
// any pointer reaches here.

#include "forman.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace hyphi_native {

void compute_forman_1d_cpu(int n_nodes, const std::int32_t* ei, const std::int32_t* ej,
                           const double* we, int n_edges, double* out) {
    if (n_edges == 0) {
        return;  // nothing to accumulate or write; empty graph is well defined
    }
    if (n_nodes <= 0) {
        throw std::runtime_error(
            "compute_forman_1d_cpu: n_nodes must be positive when n_edges > 0");
    }

    // Per-node S[v] = sum of 1/sqrt(we) over incident edges (scatter-add over both
    // endpoints). Zero-initialized; std::vector owns the storage (RAII).
    std::vector<double> s(static_cast<std::size_t>(n_nodes), 0.0);
    std::vector<double> inv_sqrt(static_cast<std::size_t>(n_edges), 0.0);

    for (int k = 0; k < n_edges; ++k) {
        const double w = we[k];
        if (!(w > 0.0)) {
            throw std::runtime_error(
                "compute_forman_1d_cpu: edge weight must be strictly positive (got a "
                "non-positive or NaN weight at edge index " + std::to_string(k) + ")");
        }
        const int u = ei[k];
        const int v = ej[k];
        if (u < 0 || u >= n_nodes || v < 0 || v >= n_nodes) {
            throw std::runtime_error(
                "compute_forman_1d_cpu: edge endpoint out of range at edge index " +
                std::to_string(k) + " (endpoints must lie in [0, n_nodes))");
        }
        const double q = 1.0 / std::sqrt(w);
        inv_sqrt[static_cast<std::size_t>(k)] = q;
        s[static_cast<std::size_t>(u)] += q;
        s[static_cast<std::size_t>(v)] += q;
    }

    // curv[k] = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]]).
    for (int k = 0; k < n_edges; ++k) {
        const double w = we[k];
        const double su = s[static_cast<std::size_t>(ei[k])];
        const double sv = s[static_cast<std::size_t>(ej[k])];
        out[k] = 4.0 - std::sqrt(w) * (su + sv);
    }
}

bool has_cuda() {
#ifdef HYPHI_NATIVE_CUDA
    return true;
#else
    return false;
#endif
}

bool has_metal() {
#ifdef HYPHI_NATIVE_METAL
    return true;
#else
    return false;
#endif
}

std::vector<double> forman_1d_dispatch(int n_nodes, const std::int32_t* ei,
                                       const std::int32_t* ej, const double* we,
                                       int n_edges, const std::string& device) {
    std::vector<double> out(static_cast<std::size_t>(n_edges < 0 ? 0 : n_edges), 0.0);

    if (device == "cpu") {
        compute_forman_1d_cpu(n_nodes, ei, ej, we, n_edges, out.data());
        return out;
    }

    if (device == "cuda") {
#ifdef HYPHI_NATIVE_CUDA
        compute_forman_1d_cuda(n_nodes, ei, ej, we, n_edges, out.data());
        return out;
#else
        throw std::runtime_error(
            "device='cuda' requested but hyphi_native was built without CUDA. Rebuild "
            "with `pip install ./native -C cmake.define.HYPHI_NATIVE_CUDA=ON` on a "
            "machine with nvcc, or use device='cpu'.");
#endif
    }

    if (device == "metal") {
#ifdef HYPHI_NATIVE_METAL
        // The metal-cpp entry takes std::size_t lengths; widen our int counts.
        compute_forman_1d_metal(static_cast<std::size_t>(n_nodes), ei, ej, we,
                                static_cast<std::size_t>(n_edges), out.data());
        return out;
#else
        throw std::runtime_error(
            "device='metal' requested but hyphi_native was built without Metal. Rebuild "
            "with `pip install ./native -C cmake.define.HYPHI_NATIVE_METAL=ON` on Apple "
            "Silicon with the Metal toolchain, or use device='cpu'.");
#endif
    }

    throw std::runtime_error(
        "unknown device '" + device + "'; expected one of {'cpu', 'cuda', 'metal'}");
}

}  // namespace hyphi_native
