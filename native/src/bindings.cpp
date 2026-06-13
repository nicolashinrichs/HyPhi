// nanobind bindings for the HyPhi native curvature core (module: hyphi_native).
//
// NumPy arrays cross the boundary zero-copy on input (nb::ndarray, buffer / DLPack
// protocol): inputs are constrained to contiguous CPU arrays of fixed dtype (int32
// endpoints, float64 weights). The output float64 array owns C++-allocated memory
// through an nb::capsule deleter (the nanobind ownership rule: never return a view
// of freed storage; always attach an owner). See NATIVE_BACKEND_DESIGN.md.
//
// Public surface (the frozen native interface):
//   has_cuda() -> bool
//   has_metal() -> bool
//   forman_1d(n_nodes, ei, ej, we, device="cpu") -> float64[E]
//
// PRECISION REGIME: CPU/CUDA float64 (exact parity, ~1e-13 vs the reference);
// Metal float32 (~1e-6 relative). The dispatcher raises on a device that was not
// compiled in or whose accelerator is absent; the Python wrapper
// (hyphi.backends.NativeExtBackend) catches and falls back to CPU.

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "forman.hpp"

namespace nb = nanobind;
using namespace hyphi_native;

namespace {

// Contiguous CPU input arrays, validated at the boundary (dtype + layout fixed by
// the type so nanobind rejects a mismatched array before we touch its data).
using EdgeIdx = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using EdgeW = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Move a std::vector into a NumPy array that owns it via a capsule deleter, so the
// returned array stays valid after this function returns (ownership rule above).
nb::ndarray<nb::numpy, double> own1d(std::vector<double>&& v) {
    const std::size_t n = v.size();
    auto* held = new std::vector<double>(std::move(v));
    nb::capsule owner(held, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });
    return nb::ndarray<nb::numpy, double>(held->data(), {n}, owner);
}

// forman_1d(n_nodes, ei, ej, we, device="cpu") -> float64[E].
nb::object py_forman_1d(int n_nodes, EdgeIdx ei, EdgeIdx ej, EdgeW we,
                        const std::string& device) {
    const std::size_t e = we.shape(0);
    if (ei.shape(0) != e || ej.shape(0) != e) {
        throw std::invalid_argument(
            "forman_1d: ei, ej, and we must have the same length E (got |ei|=" +
            std::to_string(ei.shape(0)) + ", |ej|=" + std::to_string(ej.shape(0)) +
            ", |we|=" + std::to_string(e) + ")");
    }
    if (n_nodes < 0) {
        throw std::invalid_argument("forman_1d: n_nodes must be non-negative");
    }

    // Validate at the boundary so every device (CPU, CUDA, Metal) shares one
    // contract. The CPU core checks these, but the GPU kernels do not, and an
    // out-of-range index is an out-of-bounds device write, not a clean error.
    const std::int32_t* ei_p = ei.data();
    const std::int32_t* ej_p = ej.data();
    const double* we_p = we.data();
    for (std::size_t k = 0; k < e; ++k) {
        if (ei_p[k] < 0 || ei_p[k] >= n_nodes || ej_p[k] < 0 || ej_p[k] >= n_nodes) {
            throw std::invalid_argument(
                "forman_1d: edge endpoint out of range [0, n_nodes) at edge index " +
                std::to_string(k));
        }
        if (ei_p[k] == ej_p[k]) {
            throw std::invalid_argument(
                "forman_1d: self-loops are not supported (edge index " + std::to_string(k) +
                "); drop them first to keep parity with FormanRicci on simple graphs");
        }
        if (!(we_p[k] > 0.0) || !std::isfinite(we_p[k])) {
            throw std::invalid_argument(
                "forman_1d: edge weight must be strictly positive and finite at edge index " +
                std::to_string(k));
        }
    }

    if (e > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "forman_1d: edge count exceeds the int32 limit supported by the device kernels");
    }
    const int n_edges = static_cast<int>(e);
    std::vector<double> curv = forman_1d_dispatch(
        n_nodes, ei_p, ej_p, we_p, n_edges, device);
    return nb::cast(own1d(std::move(curv)));
}

}  // namespace

NB_MODULE(hyphi_native, m) {
    m.doc() = "HyPhi native 1d Forman-Ricci curvature core (CPU float64 reference; "
              "optional CUDA float64 and Metal float32 paths). See "
              "NATIVE_BACKEND_DESIGN.md.";
    m.attr("__core_version__") = "0.1.0";

    m.def("has_cuda", &has_cuda,
          "True iff the optional CUDA path was compiled in (HYPHI_NATIVE_CUDA). This "
          "is a compile-time flag, not a runtime device probe.");

    m.def("has_metal", &has_metal,
          "True iff the optional Metal path was compiled in (HYPHI_NATIVE_METAL). This "
          "is a compile-time flag, not a runtime device probe.");

    m.def("forman_1d", &py_forman_1d, nb::arg("n_nodes"), nb::arg("ei"), nb::arg("ej"),
          nb::arg("we"), nb::arg("device") = "cpu",
          "Per-edge 1d Forman-Ricci curvature (node weights = 1).\n\n"
          "Parameters\n"
          "----------\n"
          "n_nodes : int\n"
          "    Number of graph nodes.\n"
          "ei, ej : int32 ndarray, shape (E,), C-contiguous, CPU\n"
          "    Endpoint indices of each undirected edge, values in [0, n_nodes).\n"
          "we : float64 ndarray, shape (E,), C-contiguous, CPU\n"
          "    Strictly positive edge weights.\n"
          "device : {'cpu', 'cuda', 'metal'}, default 'cpu'\n"
          "    Compute device. 'cpu' and 'cuda' compute in float64 (exact parity with "
          "the GraphRicciCurvature reference, ~1e-13); 'metal' computes in float32 "
          "(~1e-6 relative). A device not compiled in, or absent at runtime, raises a "
          "RuntimeError so the Python wrapper can fall back to CPU.\n\n"
          "Returns\n"
          "-------\n"
          "float64 ndarray, shape (E,)\n"
          "    curv[k] = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]]), where "
          "S[v] = sum of 1/sqrt(we) over edges incident to v.");
}
