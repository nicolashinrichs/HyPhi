// 1d Forman-Ricci curvature: Apple Metal host (metal-cpp).
//
// Authored by Felipe Engelberger.
//
// PRECISION REGIME: FLOAT32. Apple Metal has no general double (float64) support,
// so this backend computes in float (float32) on the GPU and widens the result to
// float64 at the boundary. Parity against the float64 CPU/CUDA reference is exact
// only to float32 tolerance: relative error on the order of 1e-6. This mirrors the
// MLX/Metal finding in code/hyphi/backends/metal_mlx.py and the frustra-webgpu
// WGSL f32 findings.
//
// REFERENCE IMPLEMENTATION: this file cannot be compiled or validated on the
// authoring machine (no Metal C++ toolchain wired up). It is written to be built
// and validated on Apple Silicon by a Metal engineer; the parity harness asserts
// the ~1e-6 relative tolerance when built.
//
// THE MATH (node weights = 1, the only notion this native path implements):
//   inv_sqrt[k] = 1 / sqrt(we[k])
//   S[v]        = sum of inv_sqrt[k] over all edges k incident to node v
//   curv[k]     = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])
//
// This host uses metal-cpp (Apple's single-header C++ bindings for Metal). The
// .metal source is embedded as a string and compiled at runtime with
// MTL::Device::newLibrary, so the build does not depend on a precompiled
// .metallib. The host:
//   1. acquires the default device (throws if none -> Python falls back to CPU),
//   2. selects the native atomic-float kernel on Metal 3.0+ devices, or the
//      atomic_uint CAS bit-cast fallback on older devices,
//   3. uploads ei/ej/we once into shared-storage MTL buffers (SoA, contiguous),
//      casting we from double to float at upload,
//   4. zero-initializes the per-node S buffer,
//   5. dispatches k_accumulate_S (or k_accumulate_S_cas), then k_edge_curv,
//   6. widens the float32 out buffer to the caller's double array.
//
// ATOMIC-FLOAT CHOICE: native atomic_float (atomic_fetch_add_explicit) is the
// primary path because it is the cleanest correct option on Metal 3.0+; the
// atomic_uint compare-and-swap bit-cast loop is the documented fallback for older
// Metal, which has no native atomic float add. Float addition is not associative,
// so accumulation order can perturb the low bits, but that is far below the
// float32 ~1e-6 tolerance this backend already accepts. See forman.metal for the
// kernel-side rationale.

#include "forman_metal.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// metal-cpp: Apple's single-header C++ bindings. Exactly one translation unit in
// the build must define the NS/MTL/CA private implementations; this is that unit.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

namespace hyphi_native {

namespace {

// The embedded Metal Shading Language source. Kept identical in spirit to
// src/metal/forman.metal; both kernels and the CAS fallback are present so the
// host can pick the variant the device supports at runtime. The build can also
// compile the standalone forman.metal into a .metallib instead; this embedded
// copy keeps the host self-contained (no on-disk resource lookup at import time).
constexpr const char* kFormanMslSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Native atomic-float scatter add (Metal 3.0+).
#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 300)
kernel void k_accumulate_S(
    device const int*    ei        [[buffer(0)]],
    device const int*    ej        [[buffer(1)]],
    device const float*  we        [[buffer(2)]],
    device atomic_float* S         [[buffer(3)]],
    constant uint&       n_edges   [[buffer(4)]],
    uint                 gid       [[thread_position_in_grid]])
{
    if (gid >= n_edges) { return; }
    const float inv_sqrt = 1.0f / sqrt(we[gid]);  // exact, matches every other backend (rsqrt is approximate)
    const int u = ei[gid];
    const int v = ej[gid];
    atomic_fetch_add_explicit(&S[u], inv_sqrt, memory_order_relaxed);
    atomic_fetch_add_explicit(&S[v], inv_sqrt, memory_order_relaxed);
}
#endif

// Portable atomic_uint compare-and-swap bit-cast float add (older Metal).
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
    }
}

kernel void k_accumulate_S_cas(
    device const int*    ei        [[buffer(0)]],
    device const int*    ej        [[buffer(1)]],
    device const float*  we        [[buffer(2)]],
    device atomic_uint*  S         [[buffer(3)]],
    constant uint&       n_edges   [[buffer(4)]],
    uint                 gid       [[thread_position_in_grid]])
{
    if (gid >= n_edges) { return; }
    const float inv_sqrt = 1.0f / sqrt(we[gid]);  // exact, matches every other backend (rsqrt is approximate)
    const int u = ei[gid];
    const int v = ej[gid];
    atomic_add_float_cas(&S[u], inv_sqrt);
    atomic_add_float_cas(&S[v], inv_sqrt);
}

// Per-edge curvature from the completed node-sum S (no atomics).
kernel void k_edge_curv(
    device const int*   ei        [[buffer(0)]],
    device const int*   ej        [[buffer(1)]],
    device const float* we        [[buffer(2)]],
    device const float* S         [[buffer(3)]],
    device float*       out       [[buffer(4)]],
    constant uint&      n_edges   [[buffer(5)]],
    uint                gid       [[thread_position_in_grid]])
{
    if (gid >= n_edges) { return; }
    const int u = ei[gid];
    const int v = ej[gid];
    out[gid] = 4.0f - sqrt(we[gid]) * (S[u] + S[v]);
}
)MSL";

// RAII guard for metal-cpp objects, which use manual retain/release semantics
// (they are not ARC-managed in C++). Releases on scope exit so no leak survives an
// early throw. metal-cpp returns +1 (owned) objects from new*/alloc-init and
// autoreleased (+0) objects from other accessors; only the owned ones go here.
template <typename T>
class MtlOwned {
public:
    MtlOwned() = default;
    explicit MtlOwned(T* p) : ptr_(p) {}
    ~MtlOwned() { if (ptr_) ptr_->release(); }
    MtlOwned(const MtlOwned&) = delete;
    MtlOwned& operator=(const MtlOwned&) = delete;
    MtlOwned(MtlOwned&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    MtlOwned& operator=(MtlOwned&& o) noexcept {
        if (this != &o) {
            if (ptr_) ptr_->release();
            ptr_ = o.ptr_;
            o.ptr_ = nullptr;
        }
        return *this;
    }
    T* get() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_ = nullptr;
};

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("hyphi_native Metal backend: " + what);
}

// True iff the device supports Metal 3.0+ (native atomic float add). Older
// devices route through the atomic_uint CAS fallback. Metal3 is the family that
// guarantees the float atomics this backend's primary kernel uses.
bool supports_metal3(MTL::Device* device) {
    return device->supportsFamily(MTL::GPUFamilyMetal3);
}

// Build a compute pipeline state for a named function in `library`.
MtlOwned<MTL::ComputePipelineState> make_pipeline(MTL::Device* device,
                                                  MTL::Library* library,
                                                  const char* fn_name) {
    NS::String* name =
        NS::String::string(fn_name, NS::StringEncoding::UTF8StringEncoding);
    MtlOwned<MTL::Function> fn(library->newFunction(name));
    if (!fn) {
        fail(std::string("kernel function '") + fn_name +
             "' not found in the compiled Metal library");
    }
    NS::Error* err = nullptr;
    MtlOwned<MTL::ComputePipelineState> pso(
        device->newComputePipelineState(fn.get(), &err));
    if (!pso) {
        std::string msg = err ? err->localizedDescription()->utf8String()
                              : "unknown error";
        fail(std::string("could not create compute pipeline for '") + fn_name +
             "': " + msg);
    }
    return pso;
}

// Allocate a shared-storage buffer of `bytes`, optionally copying `src` in.
MtlOwned<MTL::Buffer> make_buffer(MTL::Device* device, std::size_t bytes,
                                  const void* src) {
    // Metal rejects zero-length buffers; pad empty allocations to one element so
    // the n_edges == 0 case still produces valid (unused) buffers.
    const std::size_t alloc = bytes == 0 ? 1 : bytes;
    MtlOwned<MTL::Buffer> buf(
        device->newBuffer(alloc, MTL::ResourceStorageModeShared));
    if (!buf) {
        fail("failed to allocate a shared MTL buffer (" +
             std::to_string(alloc) + " bytes)");
    }
    if (src && bytes > 0) {
        std::memcpy(buf.get()->contents(), src, bytes);
    } else {
        std::memset(buf.get()->contents(), 0, alloc);
    }
    return buf;
}

// Dispatch a 1D grid of `count` threads for `pso`, with the threadgroup width
// clamped to the pipeline's maximum. Encoder buffers must already be bound.
void dispatch_1d(MTL::ComputeCommandEncoder* enc,
                 MTL::ComputePipelineState* pso, std::size_t count) {
    NS::UInteger tg = pso->maxTotalThreadsPerThreadgroup();
    if (tg == 0) tg = 1;
    if (tg > count && count > 0) tg = count;
    if (tg == 0) tg = 1;
    const MTL::Size grid = MTL::Size::Make(count == 0 ? 1 : count, 1, 1);
    const MTL::Size group = MTL::Size::Make(tg, 1, 1);
    enc->dispatchThreads(grid, group);
}

}  // namespace

bool metal_device_available() {
    MtlOwned<MTL::Device> device(MTL::CreateSystemDefaultDevice());
    return static_cast<bool>(device);
}

void compute_forman_1d_metal(std::size_t n_nodes,
                             const std::int32_t* ei,
                             const std::int32_t* ej,
                             const double* we,
                             std::size_t n_edges,
                             double* out) {
    // An empty edge set has nothing to compute; the curvature array is empty.
    if (n_edges == 0) {
        return;
    }

    // 1. Acquire the default Metal device. Null -> throw so Python falls back.
    MtlOwned<MTL::Device> device(MTL::CreateSystemDefaultDevice());
    if (!device) {
        fail("no Metal device is available "
             "(MTL::CreateSystemDefaultDevice returned null). "
             "This machine has no usable Apple GPU; use the CPU backend.");
    }

    // 2. Compile the embedded MSL source at runtime.
    NS::String* src = NS::String::string(kFormanMslSource,
                                         NS::StringEncoding::UTF8StringEncoding);
    NS::Error* lib_err = nullptr;
    MtlOwned<MTL::Library> library(
        device.get()->newLibrary(src, nullptr, &lib_err));
    if (!library) {
        std::string msg = lib_err ? lib_err->localizedDescription()->utf8String()
                                   : "unknown error";
        fail("failed to compile the embedded Metal library: " + msg);
    }

    const bool use_native = supports_metal3(device.get());
    const char* accum_fn = use_native ? "k_accumulate_S" : "k_accumulate_S_cas";

    MtlOwned<MTL::ComputePipelineState> pso_accum =
        make_pipeline(device.get(), library.get(), accum_fn);
    MtlOwned<MTL::ComputePipelineState> pso_curv =
        make_pipeline(device.get(), library.get(), "k_edge_curv");

    MtlOwned<MTL::CommandQueue> queue(device.get()->newCommandQueue());
    if (!queue) {
        fail("could not create a Metal command queue");
    }

    // 3. Upload SoA inputs once (shared storage). we is widened from the device's
    //    perspective: stored as float32, cast from the caller's double here.
    std::vector<float> we_f(n_edges);
    for (std::size_t k = 0; k < n_edges; ++k) {
        we_f[k] = static_cast<float>(we[k]);
    }

    MtlOwned<MTL::Buffer> buf_ei =
        make_buffer(device.get(), n_edges * sizeof(std::int32_t), ei);
    MtlOwned<MTL::Buffer> buf_ej =
        make_buffer(device.get(), n_edges * sizeof(std::int32_t), ej);
    MtlOwned<MTL::Buffer> buf_we =
        make_buffer(device.get(), n_edges * sizeof(float), we_f.data());

    // 4. Per-node S accumulator, zero-initialized (make_buffer zeroes when no src
    //    is given). float32 bit pattern of 0.0 is all-zero, so this is a valid
    //    start for both the native-float and the CAS (uint bits) accumulators.
    MtlOwned<MTL::Buffer> buf_S =
        make_buffer(device.get(), n_nodes * sizeof(float), nullptr);

    MtlOwned<MTL::Buffer> buf_out =
        make_buffer(device.get(), n_edges * sizeof(float), nullptr);

    const std::uint32_t n_edges_u = static_cast<std::uint32_t>(n_edges);

    // 5. Encode both dispatches into one command buffer. A compute encoder
    //    serializes its dispatches with respect to one another on Apple GPUs by
    //    default (each dispatch sees the previous one's writes), so k_edge_curv
    //    reads a fully accumulated S. Using one encoder keeps that ordering
    //    barrier implicit; the second dispatch will not begin until the first's
    //    writes are visible.
    MTL::CommandBuffer* cmd = queue.get()->commandBuffer();  // autoreleased (+0)
    if (!cmd) {
        fail("could not create a Metal command buffer");
    }
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();  // +0
    if (!enc) {
        fail("could not create a Metal compute command encoder");
    }

    // Pass 1: scatter-accumulate S.
    enc->setComputePipelineState(pso_accum.get());
    enc->setBuffer(buf_ei.get(), 0, 0);
    enc->setBuffer(buf_ej.get(), 0, 1);
    enc->setBuffer(buf_we.get(), 0, 2);
    enc->setBuffer(buf_S.get(), 0, 3);
    enc->setBytes(&n_edges_u, sizeof(n_edges_u), 4);
    dispatch_1d(enc, pso_accum.get(), n_edges);

    // Barrier so pass 2 observes pass 1's writes to S (memory + execution).
    enc->memoryBarrier(MTL::BarrierScopeBuffers);

    // Pass 2: per-edge curvature.
    enc->setComputePipelineState(pso_curv.get());
    enc->setBuffer(buf_ei.get(), 0, 0);
    enc->setBuffer(buf_ej.get(), 0, 1);
    enc->setBuffer(buf_we.get(), 0, 2);
    enc->setBuffer(buf_S.get(), 0, 3);
    enc->setBuffer(buf_out.get(), 0, 4);
    enc->setBytes(&n_edges_u, sizeof(n_edges_u), 5);
    dispatch_1d(enc, pso_curv.get(), n_edges);

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    if (cmd->status() == MTL::CommandBufferStatusError) {
        NS::Error* err = cmd->error();
        std::string msg = err ? err->localizedDescription()->utf8String()
                              : "unknown error";
        fail("Metal command buffer execution failed: " + msg);
    }

    // 6. Widen the float32 device result to the caller's float64 array.
    const float* out_f = static_cast<const float*>(buf_out.get()->contents());
    for (std::size_t k = 0; k < n_edges; ++k) {
        out[k] = static_cast<double>(out_f[k]);
    }
}

}  // namespace hyphi_native
