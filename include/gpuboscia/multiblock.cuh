#pragma once
#include <cuda_runtime.h>
// cuda::atomic_ref first ships with CUDA 11.6. Older toolkits retain
// SingleBlock support and reject explicit MultiBlock requests on the host.
#if CUDART_VERSION >= 11060
#include <cuda/atomic>
#define DRS_HAS_MULTIBLOCK_ATOMICS 1
#else
#define DRS_HAS_MULTIBLOCK_ATOMICS 0
#endif
#include <type_traits>
#include <utility>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstddef>

// All threads of all K blocks must execute node_sync() in matching order,
// including collectives inside grad_multiblock(). No grid-wide synchronization
// is permitted in user callbacks. Invalid user collectives cannot be repaired
// by cooperative launch. The launch must keep every complete group resident.
// Plain, aligned storage: zero with cudaMemsetAsync on the launch stream.
// atomic_ref does not require constructing CUDA atomic objects in raw memory.
#if DRS_HAS_MULTIBLOCK_ATOMICS
using DRSBarrierAtomic = cuda::atomic_ref<unsigned int, cuda::thread_scope_device>;
constexpr size_t drs_barrier_alignment = DRSBarrierAtomic::required_alignment;
#else
constexpr size_t drs_barrier_alignment = alignof(unsigned int);
#endif
struct alignas(16) NodeBarrier {
    alignas(drs_barrier_alignment) unsigned int arrived;
    alignas(drs_barrier_alignment) unsigned int generation;
};

class MultiBlockContext {
    NodeBarrier* barrier_;
public:
    unsigned char* lmo_workspace = nullptr;
    int node_id, block_rank, block_count;
    int thread_rank, threads_per_block, global_tid, total_threads;
    __device__ MultiBlockContext(int node, int rank, int count, NodeBarrier* barrier)
        : barrier_(barrier), node_id(node), block_rank(rank), block_count(count),
          thread_rank(threadIdx.x), threads_per_block(blockDim.x),
          global_tid(rank * blockDim.x + threadIdx.x),
          total_threads(count * blockDim.x) {}
    __device__ void block_sync() const { __syncthreads(); }
    __device__ void node_sync() const {
        // A single resident block needs no device-wide atomic protocol.
        if (block_count == 1) {
            __syncthreads();
            return;
        }
#if DRS_HAS_MULTIBLOCK_ATOMICS
        __syncthreads();
        if (threadIdx.x == 0) {
            DRSBarrierAtomic arrived(barrier_->arrived), generation(barrier_->generation);
            const unsigned int phase = generation.load(cuda::memory_order_acquire);
            // Each acq_rel RMW acquires the preceding arrival's publication.
            // The last arrival therefore transitively acquires all blocks'
            // writes, resets the counter, then publishes them via generation.
            if (arrived.fetch_add(1, cuda::memory_order_acq_rel) ==
                    static_cast<unsigned int>(block_count - 1)) {
                arrived.store(0, cuda::memory_order_relaxed);
                generation.store(phase + 1, cuda::memory_order_release);
            } else {
                while (generation.load(cuda::memory_order_acquire) == phase) {}
            }
        }
        // Publish representative's acquired writes to the whole block. A fast
        // block cannot complete the next generation until every block arrives;
        // thus wraparound cannot confuse an outstanding waiter (no ABA).
        __syncthreads();
#else
        // Unreachable through the solver: the host rejects this backend.
        asm volatile("trap;");
#endif
    }
};

enum class DRSNodeCommand : int { Stop, Gradient, AdditiveUpdate, PolicyUpdate, Gap, ForwardLMO };

struct DRSNodeControl {
    NodeBarrier barrier;
    DRSNodeCommand command;
    const double* x;
    double* gradient;
    unsigned char* objective_workspace;
    double* update_x;
    const double* direction;
    double gamma;
    double* partials;
    double gap_result;
    const double* lmo_lb;
    const double* lmo_ub;
    int lmo_n;
};

// Scratch follows all control records and reduction partials, aligned to 16 bytes.
__host__ __device__ inline size_t drs_multiblock_scratch_offset(int nodes, int k) {
    return (size_t(nodes) * (sizeof(DRSNodeControl) + size_t(k) * sizeof(double)) + 15u)
        & ~size_t(15u);
}
inline size_t drs_multiblock_control_bytes(int nodes, int k, size_t scratch_stride = 0) {
    if (nodes < 1 || k < 1) throw std::invalid_argument("invalid MultiBlock scratch dimensions");
    const size_t limit = std::numeric_limits<size_t>::max();
    const size_t per_node = sizeof(DRSNodeControl) + size_t(k) * sizeof(double);
    if (per_node > (limit - 15u) / size_t(nodes))
        throw std::invalid_argument("MultiBlock control workspace size overflows");
    const size_t base = drs_multiblock_scratch_offset(nodes, k);
    if (scratch_stride > (limit - base) / size_t(nodes))
        throw std::invalid_argument("MultiBlock LMO workspace size overflows");
    return base + size_t(nodes) * scratch_stride;
}

template<class T, class = void>
struct DRSLMOWorkspace { static constexpr size_t bytes = 0; };
template<class T>
struct DRSLMOWorkspace<T, std::void_t<decltype(T::multiblock_workspace_bytes)>> {
    static_assert(T::multiblock_workspace_bytes <= std::numeric_limits<size_t>::max() - 15u,
        "MultiBlock LMO workspace alignment overflows");
    static constexpr size_t bytes = (T::multiblock_workspace_bytes + 15u) & ~size_t(15u);
};

inline int drs_multiblock_batch_limit(int requested, int k, bool cooperative,
    int active_blocks_per_sm, int sms, int max_grid_x) {
    if (requested < 1 || k < 1)
        throw std::invalid_argument("MultiBlock batch size and blocks_per_node must be positive");
    if (!cooperative)
        throw std::runtime_error("MultiBlock requires cooperative launch support");
    const long long resident = static_cast<long long>(active_blocks_per_sm) * sms;
    const long long capacity = std::min(resident, static_cast<long long>(max_grid_x)) / k;
    if (active_blocks_per_sm < 1 || sms < 1 || capacity < 1)
        throw std::invalid_argument("MultiBlock has no resident complete node group");
    return static_cast<int>(std::min(static_cast<long long>(requested), capacity));
}

template<class T, class = void>
struct DRSHasMultiBlockWorkspace : std::false_type {};
template<class T>
struct DRSHasMultiBlockWorkspace<T, std::void_t<decltype(
    std::declval<const T&>().multiblock_workspace_bytes(0, 0))>> : std::true_type {};

template<class T, class = void>
struct DRSMultiBlockQuadratic : std::false_type {};
template<class T>
struct DRSMultiBlockQuadratic<T, std::void_t<decltype(T::supports_quadratic_exact)>>
    : std::bool_constant<T::supports_quadratic_exact> {};

template<class T, class = void>
struct DRSHasMultiBlockGradient : std::false_type {};
template<class T>
struct DRSHasMultiBlockGradient<T, std::void_t<decltype(
    std::declval<const T&>().grad_multiblock(std::declval<MultiBlockContext&>(),
        (const double*)nullptr, (double*)nullptr, (unsigned char*)nullptr))>>
    : std::true_type {};

// Optional forward oracle collective; legacy LMOs keep their block contract.
template<class T, class = void>
struct DRSHasMultiBlockLMO : std::false_type {};
template<class T>
struct DRSHasMultiBlockLMO<T, std::void_t<decltype(
    std::declval<const T&>().compute_v_multiblock(std::declval<MultiBlockContext&>(),
        0, (const double*)nullptr, (const double*)nullptr,
        (const double*)nullptr, (double*)nullptr))>> : std::true_type {};

template<class Objective, class LMO>
__device__ void drs_compute_v(const Objective&, const LMO& lmo, int tid, int n,
    const double* g, const double* lb, const double* ub, double* v) {
    lmo.compute_v(tid, n, g, lb, ub, v);
}

template<class Objective, class Algorithm>
__device__ void drs_update_partition(const Objective&, const Algorithm& alg,
    int n, double* x, const double* direction, double gamma, bool additive) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (additive) x[j] += gamma * direction[j];
        else alg.update(x, direction, gamma, j);
    }
}

// The leader executes the existing FW body, including line searches. Every
// gradient request (also trial gradients) dispatches all K blocks. Other
// Other Objective methods remain cooperative over the complete leader block.
template<class Objective>
struct DRSMultiBlockObjective {
    static constexpr int n_vars = Objective::n_vars;
    static constexpr int block_threads = Objective::block_threads;
    static constexpr int shared_bytes = Objective::shared_bytes;
    static constexpr bool supports_quadratic_exact = DRSMultiBlockQuadratic<Objective>::value;
    Objective objective;
    MultiBlockContext ctx;
    DRSNodeControl* control;
    __device__ DRSMultiBlockObjective(Objective obj, MultiBlockContext context,
                                    DRSNodeControl* state)
        : objective(obj), ctx(context), control(state) {}
    __device__ void prepare_block(int tid, unsigned char* workspace) const {
        objective.prepare_block(tid, workspace);
    }
    __device__ double value_local(int tid, const double* x, unsigned char* workspace) const {
        return objective.value_local(tid, x, workspace);
    }
    template<class T = Objective>
    __device__ auto value_from_factor_local(int tid, unsigned char* workspace) const
        -> decltype(std::declval<const T&>().value_from_factor_local(tid, workspace)) {
        return objective.value_from_factor_local(tid, workspace);
    }
    template<class T = Objective>
    __device__ auto quadratic_curvature_local(int tid, const double* direction,
                                             unsigned char* workspace) const
        -> decltype(std::declval<const T&>().quadratic_curvature_local(tid, direction, workspace)) {
        return objective.quadratic_curvature_local(tid, direction, workspace);
    }
    __device__ void grad(int, const double* x, double* gradient,
                         unsigned char* workspace) const {
        if (threadIdx.x == 0) {
            control->x = x;
            control->gradient = gradient;
            control->objective_workspace = workspace;
            control->command = DRSNodeCommand::Gradient;
        }
        ctx.node_sync();
        auto context = ctx;
        objective.grad_multiblock(context, x, gradient, workspace);
        ctx.node_sync();
    }
};

template<class Objective, class Algorithm>
__device__ void drs_update_partition(const DRSMultiBlockObjective<Objective>& obj,
    const Algorithm& alg, int n, double* x, const double* direction,
    double gamma, bool additive) {
    if (threadIdx.x == 0) {
        obj.control->update_x = x;
        obj.control->direction = direction;
        obj.control->gamma = gamma;
        obj.control->command = additive ? DRSNodeCommand::AdditiveUpdate : DRSNodeCommand::PolicyUpdate;
    }
    obj.ctx.node_sync();
    for (int j = obj.ctx.global_tid; j < n; j += obj.ctx.total_threads) {
        if (additive) x[j] += gamma * direction[j];
        else alg.update(x, direction, gamma, j);
    }
    obj.ctx.node_sync();
}

template<class Objective, class LMO>
__device__ void drs_compute_v(const DRSMultiBlockObjective<Objective>& obj,
    const LMO& lmo, int tid, int n, const double* g,
    const double* lb, const double* ub, double* v) {
    if constexpr (DRSHasMultiBlockLMO<LMO>::value) {
        if (threadIdx.x == 0) {
            obj.control->x = g;
            obj.control->lmo_lb = lb;
            obj.control->lmo_ub = ub;
            obj.control->lmo_n = n;
            obj.control->update_x = v;
            obj.control->command = DRSNodeCommand::ForwardLMO;
        }
        obj.ctx.node_sync();
        auto ctx = obj.ctx;
        lmo.compute_v_multiblock(ctx, n, g, lb, ub, v);
        obj.ctx.node_sync();
    } else {
        lmo.compute_v(tid, n, g, lb, ub, v);
    }
}
