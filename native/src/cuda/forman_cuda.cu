// forman_cuda.cu - CUDA kernels for 1d Forman-Ricci edge curvature.
//
// Authored by Felipe Engelberger.
//
// What this computes (node weights = 1, the only notion the native path
// implements). Given undirected weighted edges (ei[k], ej[k]) with weight we[k]:
//   inv_sqrt[k] = 1 / sqrt(we[k])
//   S[v]        = sum of inv_sqrt[k] over all edges k incident to node v
//   out[k]      = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])
// This is the validated closed form: it matches
// GraphRicciCurvature.FormanRicci(method="1d") to about 1e-13 in float64.
//
// Precision regime: CPU and CUDA compute in double (float64), giving exact parity
// with the reference. (Only the Metal path drops to float32, because Metal has no
// general double support; that is a separate translation unit.) Accordingly every
// device buffer and every kernel here is double precision.
//
// Status: REFERENCE IMPLEMENTATION. This translation unit is written to be
// correct-by-construction and is compiled only when the optional CUDA backend is
// enabled and nvcc is available. It has NOT been compiled or run on this machine
// (no nvcc, no NVIDIA GPU here). Build and validate on an NVIDIA GPU before
// trusting the CUDA path; the parity harness is written to run and assert once the
// extension is built with CUDA, never to fake numbers.
//
// Design notes (CUDA C++ Best Practices Guide idioms):
//   - Structure-of-arrays inputs (ei, ej, we) are contiguous and uploaded once;
//     the output is downloaded once. Host/device transfer is minimized.
//   - Device memory is held in a move-only RAII DeviceBuffer<T> (cudaMalloc in the
//     constructor, cudaFree in the destructor) so there is no naked malloc/free and
//     buffers are released even if a later cuda_check throws.
//   - The scatter-add S[v] = sum of inv_sqrt over incident edges is a histogram:
//     two edges can touch the same node concurrently, so the accumulation uses
//     atomicAdd on double. atomicAdd(double*) requires compute capability >= 6.0
//     (Pascal and newer); a build-time guard below makes that requirement explicit.
//   - Launch config: TPB = 256 threads per block, grid = (n + TPB - 1) / TPB,
//     one thread per edge in both kernels.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "forman_cuda.hpp"

// atomicAdd on double is native from compute capability 6.0 (Pascal) onward. Fail
// the build loudly on older device targets rather than silently miscompiling the
// scatter-add, which would corrupt S[v] under contention.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
#error "forman_cuda.cu requires CUDA compute capability >= 6.0 for double atomicAdd"
#endif

namespace hyphi_native {

namespace {

// Threads per block for both edge-parallel kernels (one thread per edge).
constexpr int kThreadsPerBlock = 256;

// Translate a CUDA runtime error into a C++ exception with an actionable message.
// The Python wrapper catches std::runtime_error from this path and falls back to
// the CPU implementation, so the message names the failing call to aid diagnosis.
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

// Move-only RAII device buffer. cudaMalloc in the constructor, cudaFree in the
// destructor; no copy so a device pointer is never double-freed. This guarantees
// every allocation is released even when a subsequent cuda_check throws.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        // cudaMalloc with count == 0 is implementation-defined; allocate at least
        // one element so get() is always a valid, freeable pointer.
        const std::size_t bytes = (count_ ? count_ : 1) * sizeof(T);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (ptr_) {
            // Destructor must not throw; swallow any free error (process teardown
            // or a prior fatal error already reported via cuda_check).
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return count_; }

    // Upload count_ elements from a host pointer (host to device).
    void upload(const T* host) {
        cuda_check(cudaMemcpy(ptr_, host, count_ * sizeof(T), cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
    }

    // Download count_ elements into a host pointer (device to host).
    void download(T* host) const {
        cuda_check(cudaMemcpy(host, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
    }

    // Zero-fill the whole buffer on the device (used to initialize the S array
    // before the scatter-add).
    void zero() {
        cuda_check(cudaMemset(ptr_, 0, count_ * sizeof(T)), "cudaMemset");
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace

// Kernel 1: scatter-add the per-node incidence sum S.
//   one thread per edge k; compute inv_sqrt[k] = 1 / sqrt(we[k]) and atomically add
//   it to both endpoints S[ei[k]] and S[ej[k]]. S must be zeroed before launch
//   (done host-side via DeviceBuffer::zero); the atomicAdd handles the case where
//   several edges share an endpoint concurrently. A self-loop ei == ej contributes
//   inv_sqrt twice to that node, matching the reference's "both endpoints" rule.
__global__ void k_accumulate_S(const std::int32_t* __restrict__ ei,
                               const std::int32_t* __restrict__ ej,
                               const double* __restrict__ we, int n_edges,
                               double* __restrict__ S) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_edges) return;
    const double inv_sqrt = 1.0 / sqrt(we[k]);
    atomicAdd(&S[ei[k]], inv_sqrt);
    atomicAdd(&S[ej[k]], inv_sqrt);
}

// Kernel 2: per-edge closed-form curvature.
//   one thread per edge k; out[k] = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]]).
//   Must run only after k_accumulate_S has fully populated S (the two launches are
//   ordered on the same default stream, so completion of kernel 1 is guaranteed
//   before kernel 2 reads S).
__global__ void k_edge_curv(const std::int32_t* __restrict__ ei,
                            const std::int32_t* __restrict__ ej,
                            const double* __restrict__ we,
                            const double* __restrict__ S, int n_edges,
                            double* __restrict__ out) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_edges) return;
    out[k] = 4.0 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]]);
}

// Host entry. Uploads the SoA inputs once, runs the two kernels on the default
// stream, downloads the result once. All device memory is RAII-managed.
void compute_forman_1d_cuda(int n_nodes, const std::int32_t* ei,
                            const std::int32_t* ej, const double* we, int n_edges,
                            double* out) {
    // Refuse early (with an actionable message) when no usable CUDA device exists,
    // so the Python wrapper can fall back to CPU. cudaGetDeviceCount returns an
    // error on a driver-less system; a count of zero means no GPU is present.
    int device_count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess || device_count == 0) {
        const char* reason = (count_err != cudaSuccess)
                                 ? cudaGetErrorString(count_err)
                                 : "no CUDA-capable device detected";
        throw std::runtime_error(
            std::string("CUDA Forman path requested but unavailable: ") + reason +
            ". Rebuild hyphi_native with CUDA on a host with an NVIDIA GPU, or use "
            "device=\"cpu\".");
    }

    // An edgeless graph has all-zero curvature output and nothing to launch.
    if (n_edges <= 0) return;

    const std::size_t n_e = static_cast<std::size_t>(n_edges);
    const std::size_t n_v = static_cast<std::size_t>(n_nodes > 0 ? n_nodes : 1);

    // Upload structure-of-arrays inputs once.
    DeviceBuffer<std::int32_t> d_ei(n_e);
    DeviceBuffer<std::int32_t> d_ej(n_e);
    DeviceBuffer<double> d_we(n_e);
    d_ei.upload(ei);
    d_ej.upload(ej);
    d_we.upload(we);

    // Per-node incidence sum, zeroed before the scatter-add.
    DeviceBuffer<double> d_S(n_v);
    d_S.zero();

    // Per-edge output curvature.
    DeviceBuffer<double> d_out(n_e);

    const int grid = (n_edges + kThreadsPerBlock - 1) / kThreadsPerBlock;

    // Kernel 1: accumulate S over both endpoints of every edge.
    k_accumulate_S<<<grid, kThreadsPerBlock>>>(d_ei.get(), d_ej.get(), d_we.get(),
                                               n_edges, d_S.get());
    cuda_check(cudaGetLastError(), "k_accumulate_S launch");

    // Kernel 2: closed-form curvature per edge (reads the completed S).
    k_edge_curv<<<grid, kThreadsPerBlock>>>(d_ei.get(), d_ej.get(), d_we.get(),
                                            d_S.get(), n_edges, d_out.get());
    cuda_check(cudaGetLastError(), "k_edge_curv launch");

    // Block until both kernels finish, surfacing any asynchronous launch error.
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Download the result once.
    d_out.download(out);
}

}  // namespace hyphi_native
