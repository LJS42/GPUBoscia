#pragma once
// DeviceResidentHybridSolver
//
// Runs the Frank-Wolfe relaxation for a whole B&B batch inside one CUDA
// kernel.  The host launches one long-lived kernel per batch; each CUDA block
// owns one B&B node and exits its FW loop when that node converges.
//
// DeviceObjective requirements:
//   static constexpr int n_vars;
//   static constexpr int block_threads; // maximum threads per block (warp aligned)
//   static constexpr int shared_bytes;
//   double f(const std::vector<double>& x) const;
//   __device__ double value_local(int tid, const double* x_node,
//                                 unsigned char* obj_smem) const;
//   __device__ void prepare_block(int tid, unsigned char* obj_smem) const;
//   __device__ void grad(int tid, const double* x_node,
//                        double* grad_smem,
//                        unsigned char* obj_smem) const;
//
// block_threads is a warp-aligned upper bound, not a requirement that every
// variable have its own thread. The solver launches min(round_up(n_vars),
// block_threads) threads and all cooperative methods must use block-stride
// loops when touching variable-sized arrays.
//
// grad() is cooperative over the whole CUDA block and must leave
// grad_smem[j] = grad f(x_node)[j] for j in [0, n_vars). value_local() is used
// on the device to keep the best LMO vertex; host f() is still used for the
// returned relaxation iterate and dual bound. value_local() must return the
// sum of the contributions assigned to the calling thread's block-stride
// range so it remains correct when n_vars exceeds blockDim.x.
//
// LMO requirements:
//   See lmo.cuh.  The LMO template argument is a public extension point:
//   users may pass custom CUDA-copyable LMO objects.  If the LMO provides
//   check_feasible(node, int_vars), the solver uses it before queue insertion
//   and after branching; otherwise it falls back to box feasibility.

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "algorithms.cuh"
#include "atomic_incumbent.h"
#include "bnb.h"
#include "branching.h"
#include "concurrent_queue.h"
#include "cpu_relax.h"
#include "gap_utils.h"
#include "heuristics.h"
#include "lmo.cuh"
#include "node.h"
#include "problem.h"

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
#define DRS_HAS_CUDA_ACCESS_POLICY_WINDOW 1
#else
#define DRS_HAS_CUDA_ACCESS_POLICY_WINDOW 0
#endif

struct DeviceObjectiveNoPrepare {
    __device__ void prepare_block(int /*tid*/, unsigned char* /*obj_smem*/) const {}
};

template<typename T, typename = void>
struct DRSHasL2Window : std::false_type {};

template<typename T>
struct DRSHasL2Window<T, std::void_t<
    decltype(std::declval<const T&>().l2_window_base()),
    decltype(std::declval<const T&>().l2_window_bytes())>> : std::true_type {};

// Preserve compatibility with existing custom Algorithm policies that
// predate the optional uses_dicg marker.
template<typename T, typename = void>
struct DRSUsesDICG : std::false_type {};

template<typename T>
struct DRSUsesDICG<T, std::void_t<decltype(T::uses_dicg)>>
    : std::bool_constant<T::uses_dicg> {};

template<typename T, typename = void>
struct DRSScalesDICGMaximumStep : std::false_type {};

template<typename T>
struct DRSScalesDICGMaximumStep<
    T, std::void_t<decltype(T::scales_dicg_maximum_step)>>
    : std::bool_constant<T::scales_dicg_maximum_step> {};

template<typename T, typename = void>
struct DRSSafeguardsDICGDropStep : std::false_type {};

template<typename T>
struct DRSSafeguardsDICGDropStep<
    T, std::void_t<decltype(T::safeguards_dicg_drop_step)>>
    : std::bool_constant<T::safeguards_dicg_drop_step> {};

#define DRS_CUDA_CHECK(call)                                                  \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
            throw std::runtime_error(                                         \
                std::string("CUDA ") + __FILE__ + ":" +                      \
                std::to_string(__LINE__) + " - " +                            \
                cudaGetErrorString(_e));                                      \
    } while (0)

__device__ inline double drs_warp_reduce(double v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ inline double drs_block_reduce(double v, int tid, int nw, double* ws) {
    v = drs_warp_reduce(v);
    if ((tid & 31) == 0) ws[tid >> 5] = v;
    __syncthreads();
    double r = (tid < nw) ? ws[tid] : 0.0;
    if (tid < 32) r = drs_warp_reduce(r);
    __syncthreads();
    return r;
}

__device__ inline double drs_warp_min(double v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmin(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}

__device__ inline double drs_block_min(double v, int tid, int nw, double* ws) {
    v = drs_warp_min(v);
    if ((tid & 31) == 0) ws[tid >> 5] = v;
    __syncthreads();
    double r = (tid < nw) ? ws[tid] : 1.0;
    if (tid < 32) r = drs_warp_min(r);
    __syncthreads();
    return r;
}

template<typename DeviceObjective, typename Algorithm, typename LMO>
__global__
void drs_fw_relax_batch_kernel(
    DeviceObjective obj,
    Algorithm       alg,
    LMO             lmo,
    double*         x,
    const double*   lb,
    const double*   ub,
    double*         out_gap,
    int*            out_iterations,
    uint8_t*        converged,
    double*         best_vertex,
    double*         best_vertex_val,
    double*         workspace,
    unsigned char*  objective_workspace,
    size_t          objective_workspace_stride,
    int             N,
    int             max_iters,
    const double*   eps_by_node)
{
    constexpr int n = DeviceObjective::n_vars;
    static_assert(DeviceObjective::block_threads >= 32,
                  "block_threads must allow at least one CUDA warp");
    static_assert(DeviceObjective::block_threads % 32 == 0,
                  "block_threads must be a multiple of the CUDA warp size");
    const int node = blockIdx.x;
    const int tid  = threadIdx.x;
    const int nw   = blockDim.x >> 5;
    if (node >= N) return;
    const double eps = eps_by_node[node];

    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int work_vectors =
        2 + (DRSUsesDICG<Algorithm>::value ? 1 : 0)
          + (Algorithm::uses_secant_line_search ? 2 : 0);
    double* grad = workspace + (size_t)node * work_vectors * n;
    double* v_s  = grad + n;
    double* away_s = nullptr;
    double* trial_x = nullptr;
    double* trial_grad = nullptr;
    double* ws = reinterpret_cast<double*>(smem_raw);
    double* work = v_s + n;
    if constexpr (DRSUsesDICG<Algorithm>::value) {
        away_s = work;
        work += n;
    }
    if constexpr (Algorithm::uses_secant_line_search) {
        trial_x    = work;
        trial_grad = trial_x + n;
        work        = trial_grad + n;
    }
    (void)work;
    unsigned char* obj_smem_unaligned = objective_workspace
        + (size_t)node * objective_workspace_stride;
    const uintptr_t obj_addr =
        (reinterpret_cast<uintptr_t>(obj_smem_unaligned) + 15u) & ~uintptr_t(15u);
    unsigned char* obj_smem = reinterpret_cast<unsigned char*>(obj_addr);
    __shared__ bool done_s;
    __shared__ double best_val_s;
    __shared__ double fv_s;
    __shared__ double gap_s;
    __shared__ double gamma_s;
    __shared__ double phi_lo_s;
    __shared__ double phi_hi_s;
    __shared__ double lo_s;
    __shared__ double hi_s;
    __shared__ double f_base_s;
    __shared__ double best_f_s;
    __shared__ double best_gamma_s;
    __shared__ bool line_done_s;

    double*       xn  = x  + (size_t)node * n;
    const double* lbn = lb + (size_t)node * n;
    const double* ubn = ub + (size_t)node * n;
    double*       best_vn = best_vertex + (size_t)node * n;

    if (tid == 0) {
        done_s = false;
        best_val_s = 1e300;
        best_vertex_val[node] = 1e300;
        out_iterations[node] = 0;
    }
    __syncthreads();

    obj.prepare_block(tid, obj_smem);
    __syncthreads();

    // Obtain a feasible starting vertex before asking the objective for a
    // gradient.  The bound vector lbn need not satisfy a custom LMO's set.
    for (int j = tid; j < n; j += blockDim.x)
        grad[j] = -1.0 - static_cast<double>(j) / (n + 1.0);
    __syncthreads();
    lmo.compute_v(tid, n, grad, lbn, ubn, v_s);
    __syncthreads();
    for (int j = tid; j < n; j += blockDim.x)
        xn[j] = v_s[j];
    __syncthreads();

    for (int iter = 0; iter < max_iters; iter++) {
        if (tid == 0)
            out_iterations[node] = iter + 1;
        __syncthreads();
        obj.grad(tid, xn, grad, obj_smem);
        __syncthreads();

        lmo.compute_v(tid, n, grad, lbn, ubn, v_s);
        __syncthreads();

        const double fv = drs_block_reduce(
            obj.value_local(tid, v_s, obj_smem), tid, nw, ws);
        if (tid == 0 && fv < best_val_s) {
            best_val_s = fv;
            best_vertex_val[node] = fv;
        }
        if (tid == 0)
            fv_s = fv;
        __syncthreads();
        if (fv_s == best_val_s)
            for (int j = tid; j < n; j += blockDim.x)
                best_vn[j] = v_s[j];
        __syncthreads();

        double local_gap = 0.0;
        for (int j = tid; j < n; j += blockDim.x)
            local_gap += grad[j] * (xn[j] - v_s[j]);
        const double gap = drs_block_reduce(local_gap, tid, nw, ws);

        if (tid == 0) {
            gap_s = gap;
            done_s = alg.check_converged(gap, eps);
            out_gap[node] = gap;
            converged[node] = done_s ? 1 : 0;
        }
        __syncthreads();

        if (done_s) break;

        if constexpr (Algorithm::uses_secant_line_search &&
                      !DRSUsesDICG<Algorithm>::value) {
            for (int j = tid; j < n; j += blockDim.x)
                v_s[j] -= xn[j]; // v_s now stores d = v - x.
            __syncthreads();

            const double gamma = alg.ls.search(
                obj, tid, n, nw, gap_s, 1.0, xn, v_s,
                trial_x, trial_grad, ws, obj_smem,
                &gamma_s, &phi_lo_s, &phi_hi_s, &lo_s, &hi_s,
                &f_base_s, &best_f_s, &best_gamma_s, &line_done_s);

            for (int j = tid; j < n; j += blockDim.x)
                xn[j] += gamma * v_s[j];
        } else if constexpr (DRSUsesDICG<Algorithm>::value) {
            // The in-face routine minimizes its direction.  Negating the
            // objective gradient therefore returns the away maximizer a.
            for (int j = tid; j < n; j += blockDim.x)
                grad[j] = -grad[j];
            __syncthreads();
            lmo.compute_inface_v(tid, n, grad, xn, lbn, ubn, away_s);
            __syncthreads();

            // The in-face oracle also returns a feasible extreme point, so it
            // is an incumbent candidate just like the forward LMO vertex.
            const double away_f = drs_block_reduce(
                obj.value_local(tid, away_s, obj_smem), tid, nw, ws);
            if (tid == 0 && away_f < best_val_s) {
                best_val_s = away_f;
                best_vertex_val[node] = away_f;
            }
            if (tid == 0)
                fv_s = away_f;
            __syncthreads();
            if (fv_s == best_val_s)
                for (int j = tid; j < n; j += blockDim.x)
                    best_vn[j] = away_s[j];
            __syncthreads();

            for (int j = tid; j < n; j += blockDim.x) {
                grad[j] = -grad[j];
                away_s[j] -= v_s[j]; // direction = a - v
            }
            __syncthreads();

            const double coordinate_gamma_max = drs_block_min(
                lmo.dicg_maximum_step_local(
                    tid, n, away_s, xn, lbn, ubn),
                tid, nw, ws);
            const double aggregate_gamma_max =
                lmo.dicg_sum_maximum_step(tid, n, away_s, xn);

            if (tid == 0) {
                const double gamma_max = fmax(0.0, fmin(
                    coordinate_gamma_max, aggregate_gamma_max));
                gamma_s = gamma_max;
            }
            __syncthreads();
            if constexpr (Algorithm::uses_secant_line_search) {
                // Convert a-v to the actual update direction v-a.
                for (int j = tid; j < n; j += blockDim.x)
                    away_s[j] = -away_s[j];
                __syncthreads();
                double local_pairwise_gap = 0.0;
                for (int j = tid; j < n; j += blockDim.x)
                    local_pairwise_gap -= grad[j] * away_s[j];
                const double pairwise_gap = drs_block_reduce(
                    local_pairwise_gap, tid, nw, ws);
                const double gamma = alg.ls.search(
                    obj, tid, n, nw, pairwise_gap, gamma_s, xn, away_s,
                    trial_x, trial_grad, ws, obj_smem,
                    &gamma_s, &phi_lo_s, &phi_hi_s, &lo_s, &hi_s,
                    &f_base_s, &best_f_s, &best_gamma_s, &line_done_s);
                for (int j = tid; j < n; j += blockDim.x)
                    xn[j] += gamma * away_s[j];
            } else {
                if (tid == 0) {
                    const double gamma_max = gamma_s;
                    const double open_loop_step =
                        alg.step_size(iter, gap_s);
                    if constexpr (DRSSafeguardsDICGDropStep<Algorithm>::value) {
                        gamma_s = gamma_max <= open_loop_step
                            ? gamma_max
                            : gamma_max * open_loop_step;
                    } else if constexpr (
                        DRSScalesDICGMaximumStep<Algorithm>::value) {
                        gamma_s = gamma_max * open_loop_step;
                    } else {
                        gamma_s = fmin(gamma_max, open_loop_step);
                    }
                }
                __syncthreads();
                for (int j = tid; j < n; j += blockDim.x)
                    alg.update(xn, away_s, gamma_s, j);
            }
        } else {
            const double gamma = alg.step_size(iter, gap_s);
            for (int j = tid; j < n; j += blockDim.x)
                alg.update(xn, v_s, gamma, j);
        }
        __syncthreads();
    }

    // The last FW step may have changed xn after gap_s was computed.  This is
    // especially visible with secant line search, where a large final step can
    // make fx and the stale pre-step gap belong to very different points.
    // Recompute the certificate at the returned iterate so the host-side
    // dual bound fx - gap is coherent.
    obj.grad(tid, xn, grad, obj_smem);
    __syncthreads();

    lmo.compute_v(tid, n, grad, lbn, ubn, v_s);
    __syncthreads();

    const double final_fv = drs_block_reduce(
        obj.value_local(tid, v_s, obj_smem), tid, nw, ws);
    if (tid == 0 && final_fv < best_val_s) {
        best_val_s = final_fv;
        best_vertex_val[node] = final_fv;
    }
    if (tid == 0)
        fv_s = final_fv;
    __syncthreads();
    if (fv_s == best_val_s)
        for (int j = tid; j < n; j += blockDim.x)
            best_vn[j] = v_s[j];
    __syncthreads();

    double final_local_gap = 0.0;
    for (int j = tid; j < n; j += blockDim.x)
        final_local_gap += grad[j] * (xn[j] - v_s[j]);
    const double final_gap = drs_block_reduce(final_local_gap, tid, nw, ws);
    if (tid == 0) {
        out_gap[node] = final_gap;
        converged[node] = alg.check_converged(final_gap, eps) ? 1 : 0;
    }
}

struct DeviceResidentConfig {
    int    gpu_batch_size = 1024;
    int    num_streams    = 8;
    // 0 selects a warp-aligned size automatically. A positive override must
    // be a multiple of the CUDA warp size (32).
    int    gpu_block_threads = 0;
    int    gpu_max_iters  = 20000;
    double gpu_eps        = 1e-3;
    double gpu_eps_decay  = 0.8;
    double gpu_eps_min    = 1e-6;
    int    serial_iters   = 10000;
    double serial_eps     = 1e-5;
    double integer_tol    = 1e-6;
    double abs_gap_tol    = 1e-6;
    double rel_gap_tol    = 1e-2;
};

enum class DRSSchedulingEventType {
    Dispatched,
    CompletionDetached,
    PostprocessStarted
};

struct DRSSchedulingEvent {
    DRSSchedulingEventType type;
    int                    slot_id;
    std::uint64_t          generation;
    int                    n_nodes;
};

struct DRSCompletionSignal {
    std::mutex*              mtx;
    std::condition_variable* cv;
    std::queue<int>*         q;
    int                      slot_id;
};

static void CUDART_CB drs_signal_cb(void* raw) {
    auto* s = static_cast<DRSCompletionSignal*>(raw);
    { std::lock_guard<std::mutex> lk(*s->mtx); s->q->push(s->slot_id); }
    s->cv->notify_one();
}

struct DRSStreamSlot {
    int          id       = -1;
    cudaStream_t stream   = nullptr;
    int          capacity = 0;
    int          n_nodes  = 0;
    bool         in_use   = false;
    std::uint64_t generation = 0;

    DRSCompletionSignal sig{};
    std::vector<Node>   batch_nodes;
    std::thread         slot_thread;

    double*  h_x   = nullptr;
    double*  h_lb  = nullptr;
    double*  h_ub  = nullptr;
    double*  h_gap = nullptr;
    double*  h_eps = nullptr;
    int*     h_iterations = nullptr;
    double*  h_best_vertex = nullptr;
    double*  h_best_vertex_val = nullptr;
    uint8_t* h_cnv = nullptr;

    double*  d_x   = nullptr;
    double*  d_lb  = nullptr;
    double*  d_ub  = nullptr;
    double*  d_gap = nullptr;
    double*  d_eps = nullptr;
    int*     d_iterations = nullptr;
    double*  d_best_vertex = nullptr;
    double*  d_best_vertex_val = nullptr;
    double*  d_workspace = nullptr;
    unsigned char* d_objective_workspace = nullptr;
    uint8_t* d_cnv = nullptr;

    void alloc(int cap, int n, int work_vectors, size_t objective_stride) {
        capacity = cap;
        const size_t xn = (size_t)cap * n;

        DRS_CUDA_CHECK(cudaHostAlloc(&h_x,   xn  * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_lb,  xn  * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_ub,  xn  * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_gap, cap * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_eps, cap * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_iterations, cap * sizeof(int), cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_best_vertex,     xn  * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_best_vertex_val, cap * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_cnv, cap * sizeof(uint8_t), cudaHostAllocDefault));

        DRS_CUDA_CHECK(cudaMalloc(&d_x,   xn  * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_lb,  xn  * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_ub,  xn  * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_gap, cap * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_eps, cap * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_iterations, cap * sizeof(int)));
        DRS_CUDA_CHECK(cudaMalloc(&d_best_vertex,     xn  * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_best_vertex_val, cap * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_cnv, cap * sizeof(uint8_t)));
        DRS_CUDA_CHECK(cudaMalloc(
            &d_workspace, xn * (size_t)work_vectors * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(
            &d_objective_workspace, (size_t)cap * objective_stride));

        DRS_CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void free_all() {
        if (slot_thread.joinable()) slot_thread.join();
        cudaFreeHost(h_x);
        cudaFreeHost(h_lb);
        cudaFreeHost(h_ub);
        cudaFreeHost(h_gap);
        cudaFreeHost(h_eps);
        cudaFreeHost(h_iterations);
        cudaFreeHost(h_best_vertex);
        cudaFreeHost(h_best_vertex_val);
        cudaFreeHost(h_cnv);
        cudaFree(d_x);
        cudaFree(d_lb);
        cudaFree(d_ub);
        cudaFree(d_gap);
        cudaFree(d_eps);
        cudaFree(d_iterations);
        cudaFree(d_best_vertex);
        cudaFree(d_best_vertex_val);
        cudaFree(d_cnv);
        cudaFree(d_workspace);
        cudaFree(d_objective_workspace);
        if (stream) cudaStreamDestroy(stream);
    }
};

// Host-owned snapshot of a completed GPU batch.  A completion is detached
// from its stream slot before CPU post-processing so that the slot can be
// refilled immediately without overwriting results that are still needed for
// certificate validation, heuristics, pruning, and branching.
struct DRSCompletedBatch {
    int                 n_nodes = 0;
    int                 slot_id = -1;
    std::uint64_t       generation = 0;
    std::vector<Node>   nodes;
    std::vector<double> x;
    std::vector<int>    iterations;
    std::vector<double> best_vertex;
    std::vector<double> best_vertex_val;
};

template<typename DeviceObjective, typename Algorithm, typename LMO = BoxLMO>
class DeviceResidentHybridSolver {
public:
    DeviceResidentHybridSolver(
        const Problem&       prob,
        DeviceObjective      obj,
        Algorithm            alg,
        BranchStrategy       branch_sel,
        DeviceResidentConfig cfg = {},
        LMO                  lmo = {},
        double               time_limit_seconds =
            std::numeric_limits<double>::infinity())
        : prob_(prob), obj_(obj), alg_(alg), lmo_(lmo)
        , branch_sel_(std::move(branch_sel)), cfg_(cfg)
        , time_limit_seconds_(time_limit_seconds)
    {
        if (!std::isfinite(cfg_.gpu_eps) || cfg_.gpu_eps <= 0.0
                || !std::isfinite(cfg_.gpu_eps_min)
                || cfg_.gpu_eps_min <= 0.0
                || !std::isfinite(cfg_.gpu_eps_decay)
                || cfg_.gpu_eps_decay <= 0.0 || cfg_.gpu_eps_decay > 1.0
                || !std::isfinite(cfg_.abs_gap_tol) || cfg_.abs_gap_tol < 0.0
                || !std::isfinite(cfg_.rel_gap_tol) || cfg_.rel_gap_tol < 0.0)
            throw std::invalid_argument("invalid tolerance configuration");
        if (std::isnan(time_limit_seconds_) || time_limit_seconds_ <= 0.0)
            throw std::invalid_argument("time limit must be positive");
        if (prob_.n_vars != DeviceObjective::n_vars)
            throw std::runtime_error("DeviceObjective::n_vars does not match Problem::n_vars");
        configure_kernel_shared_memory();
        slots_.resize(cfg_.num_streams);
        for (int i = 0; i < cfg_.num_streams; i++) {
            slots_[i].id = i;
            slots_[i].alloc(
                cfg_.gpu_batch_size, prob_.n_vars,
                kernel_work_vectors(), objective_workspace_stride());
            slots_[i].sig = { &completed_mtx_, &completed_cv_, &completed_, i };
            configure_l2_window(slots_[i].stream);
        }
    }

    ~DeviceResidentHybridSolver() {
        for (auto& slot : slots_)
            slot.free_all();
    }

    void push(const Node& nd) {
        if (lmo_check_feasible(lmo_, nd, prob_.int_vars))
            queue_.push(nd);
    }

    void set_progress_callback(
        std::function<void(int, int, double, double, bool)> cb)
    {
        progress_cb_ = std::move(cb);
    }

    // Optional deterministic scheduling observer, primarily useful for
    // integration tests and timeline instrumentation.  It is invoked by the
    // solver's host worker thread and must not call back into the solver.
    void set_scheduling_callback(
        std::function<void(const DRSSchedulingEvent&)> cb)
    {
        scheduling_cb_ = std::move(cb);
    }

    void solve() {
        solve_started_ = Clock::now();
        gpu_worker();
        DRS_CUDA_CHECK(cudaDeviceSynchronize());
        if (final_dual_bound_ == -std::numeric_limits<double>::infinity())
            final_dual_bound_ = queue_.best_bound(incumbent_.get());
    }

    double              incumbent_value() const { return incumbent_.get(); }
    std::vector<double> incumbent_x()    const { return incumbent_.get_x(); }
    int                 nodes_processed() const { return nodes_done_.load(); }
    double              dual_bound()      const { return final_dual_bound_; }
    int                 last_gpu_iterations() const { return last_gpu_iterations_.load(); }
    double              last_gpu_gap() const { return last_gpu_gap_.load(); }
    bool                last_gpu_converged() const { return last_gpu_converged_.load(); }
    bool                reached_tolerance() const { return reached_tolerance_; }
    bool                reached_time_limit() const { return reached_time_limit_; }

private:
    const Problem&       prob_;
    DeviceObjective      obj_;
    Algorithm            alg_;
    LMO                  lmo_;
    BranchStrategy       branch_sel_;
    DeviceResidentConfig cfg_;

    ConcurrentQueue queue_;
    ConcurrentBoundSet active_bounds_;
    AtomicIncumbent incumbent_;
    std::atomic<int> active_tasks_{0};
    std::atomic<int> next_id_{1};
    std::atomic<int> nodes_done_{0};
    std::atomic<int> last_gpu_iterations_{0};
    std::atomic<double> last_gpu_gap_{std::numeric_limits<double>::infinity()};
    std::atomic<bool> last_gpu_converged_{false};
    std::atomic<bool> done_{false};
    double final_dual_bound_ = -std::numeric_limits<double>::infinity();
    bool reached_tolerance_ = false;
    bool reached_time_limit_ = false;
    using Clock = std::chrono::steady_clock;
    Clock::time_point solve_started_{};
    double time_limit_seconds_ = std::numeric_limits<double>::infinity();
    std::function<void(int, int, double, double, bool)> progress_cb_;
    std::function<void(const DRSSchedulingEvent&)> scheduling_cb_;

    std::vector<DRSStreamSlot> slots_;
    std::mutex                 completed_mtx_;
    std::condition_variable    completed_cv_;
    std::queue<int>            completed_;

    static constexpr int kernel_work_vectors() {
        return
            2 + (DRSUsesDICG<Algorithm>::value ? 1 : 0)
              + (Algorithm::uses_secant_line_search ? 2 : 0);
    }

    static constexpr size_t objective_workspace_stride() {
        constexpr size_t bytes = DeviceObjective::shared_bytes > 0
            ? (size_t)DeviceObjective::shared_bytes : size_t(16);
        return (bytes + 15u) & ~size_t(15u);
    }

    static constexpr int kernel_dynamic_shared_bytes() {
        constexpr int nw = DeviceObjective::block_threads / 32;
        return nw * (int)sizeof(double);
    }

    int kernel_block_threads(int n) const {
        cudaFuncAttributes attrs{};
        DRS_CUDA_CHECK(cudaFuncGetAttributes(
            &attrs, drs_fw_relax_batch_kernel<DeviceObjective, Algorithm, LMO>));
        const int limit = std::min(
            DeviceObjective::block_threads, attrs.maxThreadsPerBlock);
        const int warp_limit = (limit / 32) * 32;
        if (warp_limit < 32)
            throw std::runtime_error("CUDA kernel cannot launch a full warp");
        if (cfg_.gpu_block_threads != 0) {
            if (cfg_.gpu_block_threads < 32
                    || cfg_.gpu_block_threads % 32 != 0)
                throw std::invalid_argument(
                    "gpu_block_threads must be zero or a positive multiple of 32");
            if (cfg_.gpu_block_threads > warp_limit)
                throw std::invalid_argument(
                    "gpu_block_threads exceeds the objective/kernel/device limit");
            return cfg_.gpu_block_threads;
        }
        const int wanted = ((n + 31) / 32) * 32;
        return std::min(std::max(32, wanted), warp_limit);
    }

    void configure_kernel_shared_memory() const {
        constexpr int dynamic_bytes = kernel_dynamic_shared_bytes();

        int device = 0;
        int default_limit = 0;
        int optin_limit = 0;
        cudaFuncAttributes attrs{};
        DRS_CUDA_CHECK(cudaGetDevice(&device));
        DRS_CUDA_CHECK(cudaDeviceGetAttribute(
            &default_limit, cudaDevAttrMaxSharedMemoryPerBlock, device));
        DRS_CUDA_CHECK(cudaDeviceGetAttribute(
            &optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        DRS_CUDA_CHECK(cudaFuncGetAttributes(
            &attrs,
            drs_fw_relax_batch_kernel<DeviceObjective, Algorithm, LMO>));

        const size_t total_bytes =
            (size_t)attrs.sharedSizeBytes + (size_t)dynamic_bytes;
        if (total_bytes > (size_t)optin_limit) {
            throw std::runtime_error(
                "CUDA kernel requires " + std::to_string(total_bytes) +
                " bytes of shared memory per block (" +
                std::to_string(dynamic_bytes) + " dynamic + " +
                std::to_string(attrs.sharedSizeBytes) +
                " static), but this device supports only " +
                std::to_string(optin_limit) + " opt-in bytes");
        }

        if (total_bytes > (size_t)default_limit) {
            DRS_CUDA_CHECK(cudaFuncSetAttribute(
                drs_fw_relax_batch_kernel<DeviceObjective, Algorithm, LMO>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                dynamic_bytes));
        }
    }

    void configure_l2_window(cudaStream_t stream) const {
#if DRS_HAS_CUDA_ACCESS_POLICY_WINDOW
        if constexpr (DRSHasL2Window<DeviceObjective>::value) {
            const void* base = obj_.l2_window_base();
            size_t bytes = obj_.l2_window_bytes();
            if (!base || bytes == 0) return;

            int device = 0;
            DRS_CUDA_CHECK(cudaGetDevice(&device));

            cudaDeviceProp prop{};
            cudaError_t attr_err = cudaGetDeviceProperties(&prop, device);
            if (attr_err != cudaSuccess || prop.accessPolicyMaxWindowSize <= 0) {
                cudaGetLastError();
                return;
            }

            cudaStreamAttrValue attr{};
            attr.accessPolicyWindow.base_ptr  = const_cast<void*>(base);
            attr.accessPolicyWindow.num_bytes =
                std::min(bytes, (size_t)prop.accessPolicyMaxWindowSize);
            attr.accessPolicyWindow.hitRatio  = 0.8;
            attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

            cudaError_t set_err = cudaStreamSetAttribute(
                stream, cudaStreamAttributeAccessPolicyWindow, &attr);
            if (set_err != cudaSuccess)
                cudaGetLastError();
        }
#else
        (void)stream;
#endif
    }

    void gpu_worker() {
        while (!done_.load(std::memory_order_relaxed)) {
            if (finish_if_time_limit_reached())
                return;
            std::vector<int> ready;
            {
                std::lock_guard<std::mutex> lk(completed_mtx_);
                while (!completed_.empty()) {
                    ready.push_back(completed_.front());
                    completed_.pop();
                }
            }

            std::vector<DRSCompletedBatch> completed_batches;
            completed_batches.reserve(ready.size());
            for (int sid : ready)
                completed_batches.push_back(detach_completed_slot(sid));

            // Refill every newly released slot before doing CPU-side work on
            // the completed batches.  Their results now live in independent
            // host storage and cannot be overwritten by the new launches.
            fill_free_slots();

            for (auto& batch : completed_batches)
                process_completed_batch(batch);

            if (active_tasks_.load(std::memory_order_acquire) == 0
                    && finish_if_search_complete())
                return;

            // Processing a completed batch may have generated enough child
            // nodes to fill slots that could not be refilled above.
            fill_free_slots();

            bool gpu_active = false;
            bool free_slot = false;
            for (auto& slot : slots_) {
                if (slot.in_use) {
                    gpu_active = true;
                } else {
                    free_slot = true;
                }
            }

            // Never leave the GPU completely idle just because the frontier
            // is smaller than a full batch. This bootstraps the search (and
            // drains its tail) with a partial batch. While another stream is
            // active, the serial path below keeps expanding nodes in an
            // attempt to fill the next batch.
            if (!gpu_active) {
                fill_free_slots(true);
                for (auto& slot : slots_)
                    if (slot.in_use) {
                        gpu_active = true;
                        break;
                    }
                if (gpu_active)
                    continue;
            }

            // While a stream is free, use the CPU to expand one queued node
            // rather than launching a short GPU batch. This can overlap batch
            // formation with work already running on other streams.
            auto serial_batch = free_slot
                ? queue_.pop_batch(1, incumbent_.get())
                : std::vector<Node>{};
            if (!serial_batch.empty()) {
                active_bounds_.insert(serial_batch);
                active_tasks_.fetch_add(1, std::memory_order_relaxed);
                run_serial_node(serial_batch[0], serial_batch[0].dual_bound);
                if (done_.load(std::memory_order_relaxed))
                    return;
                fill_free_slots();
            } else if (gpu_active) {
                std::unique_lock<std::mutex> lk(completed_mtx_);
                completed_cv_.wait_for(lk, std::chrono::microseconds(200),
                    [&] {
                        return !completed_.empty()
                            || done_.load(std::memory_order_relaxed);
                    });
            } else {
                if (active_tasks_.load(std::memory_order_acquire) == 0
                        && finish_if_search_complete())
                    return;
                std::this_thread::yield();
            }
        }

        std::vector<int> remaining;
        {
            std::lock_guard<std::mutex> lk(completed_mtx_);
            while (!completed_.empty()) {
                remaining.push_back(completed_.front());
                completed_.pop();
            }
        }
        for (int sid : remaining) {
            auto batch = detach_completed_slot(sid);
            process_completed_batch(batch);
        }
    }

    void fill_free_slots(bool allow_partial = false) {
        for (auto& slot : slots_) {
            if (slot.in_use) continue;
            // Normally launch only full batches. The caller may allow one
            // partial batch when every GPU stream would otherwise be idle.
            auto batch = queue_.pop_full_batch(
                cfg_.gpu_batch_size, incumbent_.get());
            if (batch.empty() && allow_partial)
                batch = queue_.pop_batch(
                    cfg_.gpu_batch_size, incumbent_.get());
            if (batch.empty()) break;
            active_bounds_.insert(batch);
            active_tasks_.fetch_add((int)batch.size(), std::memory_order_relaxed);
            submit_slot(slot, std::move(batch));
            // At most one under-filled batch should be launched. Once it has
            // consumed the frontier there cannot be useful work for another
            // free slot in this pass.
            if (allow_partial) break;
        }
    }

    void run_serial_node(Node nd, double tracked_active_bound) {
        const double inherited_bound = nd.dual_bound;
        const double eps = node_tolerance(nd);
        auto res = cpu_frank_wolfe(prob_, nd, lmo_, cfg_.serial_iters, eps);
        nd.dual_bound = std::max(inherited_bound, res.dual_bound);

        bool new_incumbent = false;
        if (!res.best_vertex.empty())
            new_incumbent = incumbent_.try_update(res.best_vertex_val, res.best_vertex);
        try_rounded_incumbent(res.x, nd, new_incumbent);

        if (!should_prune(nd)) {
            if (is_integer_feasible(res.x, prob_.int_vars, cfg_.integer_tol)) {
                // The rounded heuristic above already tested the strict
                // integer candidate. Do not store the near-integer raw point.
            } else {
                int nid = next_id_.fetch_add(8, std::memory_order_relaxed);
                for (auto& child : branch_sel_(nd, nid, res.x, prob_.int_vars))
                    if (lmo_check_feasible(lmo_, child, prob_.int_vars) && !should_prune(child))
                        queue_.push(child);
            }
        }

        const int iter = nodes_done_.fetch_add(1, std::memory_order_relaxed) + 1;
        active_tasks_.fetch_sub(1, std::memory_order_acq_rel);
        active_bounds_.erase_one(tracked_active_bound);
        emit_progress(iter, new_incumbent);
        finish_if_search_complete();
    }

    void submit_slot(DRSStreamSlot& slot, std::vector<Node> batch) {
        const int n = prob_.n_vars;
        const int N = (int)batch.size();
        slot.batch_nodes = std::move(batch);
        slot.n_nodes = N;
        slot.in_use = true;
        ++slot.generation;

        for (int i = 0; i < N; i++) {
            const Node& nd = slot.batch_nodes[i];
            for (int j = 0; j < n; j++) {
                slot.h_x [i*n+j] = nd.lb[j];
                slot.h_lb[i*n+j] = nd.lb[j];
                slot.h_ub[i*n+j] = nd.ub[j];
            }
            slot.h_gap[i] = 0.0;
            slot.h_eps[i] = node_tolerance(nd);
            slot.h_iterations[i] = 0;
            slot.h_best_vertex_val[i] = 1e300;
            slot.h_cnv[i] = 0;
        }

        if (slot.slot_thread.joinable())
            slot.slot_thread.join();

        slot.slot_thread = std::thread([this, &slot, N, n]() {
            run_slot(slot, N, n);
        });

        emit_scheduling_event(
            DRSSchedulingEventType::Dispatched,
            slot.id, slot.generation, N);
    }

    void run_slot(DRSStreamSlot& slot, int N, int n) {
        const size_t xn = (size_t)N * n;

        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.d_x,   slot.h_x,   xn * sizeof(double),  cudaMemcpyHostToDevice, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.d_lb,  slot.h_lb,  xn * sizeof(double),  cudaMemcpyHostToDevice, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.d_ub,  slot.h_ub,  xn * sizeof(double),  cudaMemcpyHostToDevice, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.d_gap, slot.h_gap, N  * sizeof(double),  cudaMemcpyHostToDevice, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.d_eps, slot.h_eps, N  * sizeof(double),  cudaMemcpyHostToDevice, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.d_iterations, slot.h_iterations, N * sizeof(int), cudaMemcpyHostToDevice, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.d_best_vertex_val, slot.h_best_vertex_val, N * sizeof(double), cudaMemcpyHostToDevice, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.d_cnv, slot.h_cnv, N  * sizeof(uint8_t), cudaMemcpyHostToDevice, slot.stream));

        const int block_threads = kernel_block_threads(n);
        constexpr int smem = kernel_dynamic_shared_bytes();

        drs_fw_relax_batch_kernel<DeviceObjective, Algorithm, LMO>
            <<<N, block_threads, smem, slot.stream>>>(
                obj_, alg_, lmo_,
                slot.d_x, slot.d_lb, slot.d_ub, slot.d_gap,
                slot.d_iterations, slot.d_cnv,
                slot.d_best_vertex, slot.d_best_vertex_val,
                slot.d_workspace, slot.d_objective_workspace,
                objective_workspace_stride(),
                N, cfg_.gpu_max_iters, slot.d_eps);
        DRS_CUDA_CHECK(cudaGetLastError());

        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.h_x,   slot.d_x,   xn * sizeof(double),  cudaMemcpyDeviceToHost, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.h_gap, slot.d_gap, N  * sizeof(double),  cudaMemcpyDeviceToHost, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.h_iterations, slot.d_iterations, N * sizeof(int), cudaMemcpyDeviceToHost, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.h_best_vertex, slot.d_best_vertex, xn * sizeof(double), cudaMemcpyDeviceToHost, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.h_best_vertex_val, slot.d_best_vertex_val, N * sizeof(double), cudaMemcpyDeviceToHost, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.h_cnv, slot.d_cnv, N  * sizeof(uint8_t), cudaMemcpyDeviceToHost, slot.stream));
        DRS_CUDA_CHECK(cudaLaunchHostFunc(slot.stream, drs_signal_cb, &slot.sig));
    }

    DRSCompletedBatch detach_completed_slot(int sid) {
        DRSStreamSlot& slot = slots_[sid];
        const int n = prob_.n_vars;
        const int N = slot.n_nodes;

        // The CUDA completion callback is queued after all device-to-host
        // copies, so reaching here means the slot buffers are complete.  Join
        // the launcher thread as well before making the slot reusable.
        if (slot.slot_thread.joinable())
            slot.slot_thread.join();

        DRSCompletedBatch batch;
        batch.n_nodes = N;
        batch.slot_id = sid;
        batch.generation = slot.generation;
        batch.nodes = std::move(slot.batch_nodes);
        batch.x.assign(slot.h_x, slot.h_x + (size_t)N*n);
        batch.iterations.assign(slot.h_iterations, slot.h_iterations + N);
        batch.best_vertex.assign(
            slot.h_best_vertex, slot.h_best_vertex + (size_t)N*n);
        batch.best_vertex_val.assign(
            slot.h_best_vertex_val, slot.h_best_vertex_val + N);

        slot.n_nodes = 0;
        slot.in_use = false;
        emit_scheduling_event(
            DRSSchedulingEventType::CompletionDetached,
            sid, batch.generation, N);
        return batch;
    }

    void process_completed_batch(DRSCompletedBatch& batch) {
        const int n = prob_.n_vars;
        const int N = batch.n_nodes;

        emit_scheduling_event(
            DRSSchedulingEventType::PostprocessStarted,
            batch.slot_id, batch.generation, N);

        for (int i = 0; i < N; i++) {
            if (finish_if_time_limit_reached())
                return;
            Node nd = batch.nodes[i];
            const double active_bound = nd.dual_bound;
            std::vector<double> xi(batch.x.begin() + (size_t)i*n,
                                   batch.x.begin() + (size_t)(i+1)*n);

            const double fx = obj_.f(xi);
            // Recompute the FW certificate on the host before using a GPU
            // result to prune or close a node.  In particular, an integral
            // last iterate is not sufficient evidence that the relaxation
            // was solved when the device iteration limit was reached.
            std::vector<double> host_grad(n, 0.0);
            std::vector<double> host_vertex(n, 0.0);
            prob_.grad_f(xi, host_grad);
            lmo_compute_v_cpu(lmo_, host_grad, nd.lb, nd.ub, host_vertex);
            double gap = 0.0;
            for (int j = 0; j < n; j++)
                gap += host_grad[j] * (xi[j] - host_vertex[j]);

            last_gpu_iterations_.store(batch.iterations[i], std::memory_order_relaxed);
            last_gpu_gap_.store(gap, std::memory_order_relaxed);
            last_gpu_converged_.store(
                std::isfinite(gap) && gap >= 0.0
                    && gap < node_tolerance(nd),
                std::memory_order_relaxed);

            nd.dual_bound = fastbb::certified_fw_lower_bound(
                active_bound, fx, gap);

            bool new_incumbent = false;
            if (batch.best_vertex_val[i] < 1e299) {
                std::vector<double> best_vertex(
                    batch.best_vertex.begin() + (size_t)i*n,
                    batch.best_vertex.begin() + (size_t)(i+1)*n);
                new_incumbent =
                    incumbent_.try_update(batch.best_vertex_val[i], best_vertex);
            }
            try_rounded_incumbent(xi, nd, new_incumbent);

            const bool certificate_valid =
                std::isfinite(gap) && gap >= 0.0
                && gap < node_tolerance(nd);

            if (should_prune(nd)) {
                const int iter = nodes_done_.fetch_add(1, std::memory_order_relaxed) + 1;
                active_tasks_.fetch_sub(1, std::memory_order_acq_rel);
                active_bounds_.erase_one(active_bound);
                emit_progress(iter, new_incumbent);
                finish_if_search_complete();
                continue;
            }

            if (is_integer_feasible(xi, prob_.int_vars, cfg_.integer_tol)
                    && !certificate_valid) {
                // An integral iterate cannot be branched, but without a
                // valid FW certificate it also cannot safely close the node.
                // Use the serial refinement path for this exceptional case;
                // fractional GPU iterates remain eligible for branching.
                run_serial_node(nd, active_bound);
                continue;
            }

            if (done_.load(std::memory_order_relaxed)) {
                // Stop expanding this completed batch once another path has
                // proven the search complete.
            } else if (is_integer_feasible(xi, prob_.int_vars, cfg_.integer_tol)) {
                // The rounded heuristic above already tested the strict
                // integer candidate. Do not store the near-integer raw point.
            } else {
                int nid = next_id_.fetch_add(8, std::memory_order_relaxed);
                for (auto& child : branch_sel_(nd, nid, xi, prob_.int_vars))
                    if (lmo_check_feasible(lmo_, child, prob_.int_vars) && !should_prune(child))
                        queue_.push(child);
            }

            const int iter = nodes_done_.fetch_add(1, std::memory_order_relaxed) + 1;
            active_tasks_.fetch_sub(1, std::memory_order_acq_rel);
            active_bounds_.erase_one(active_bound);
            emit_progress(iter, new_incumbent);
            finish_if_search_complete();
        }

        if (active_tasks_.load(std::memory_order_acquire) == 0
                && finish_if_search_complete())
            return;
    }

    void signal_done() {
        done_.store(true, std::memory_order_release);
        completed_cv_.notify_all();
    }

    void emit_scheduling_event(
        DRSSchedulingEventType type,
        int slot_id,
        std::uint64_t generation,
        int n_nodes)
    {
        if (scheduling_cb_)
            scheduling_cb_({type, slot_id, generation, n_nodes});
    }

    bool finish_if_time_limit_reached() {
        if (!std::isfinite(time_limit_seconds_))
            return false;
        const double elapsed = std::chrono::duration<double>(
            Clock::now() - solve_started_).count();
        if (elapsed < time_limit_seconds_)
            return false;

        final_dual_bound_ = global_live_bound(incumbent_.get());
        reached_time_limit_ = true;
        signal_done();
        return true;
    }

    double global_live_bound(double incumbent) const {
        const double queue_bound = queue_.best_bound(incumbent);
        const double active_bound = active_bounds_.best_bound(incumbent);
        return std::min(queue_bound, active_bound);
    }

    bool finish_if_search_complete() {
        if (done_.load(std::memory_order_acquire))
            return true;

        // An empty queue is only proof of optimality after all in-flight
        // nodes have completed.  The gap test, however, is valid at any time:
        // global_live_bound() includes both queued and active node bounds.
        if (queue_.empty()
                && active_tasks_.load(std::memory_order_acquire) == 0) {
            final_dual_bound_ = incumbent_.get();
            signal_done();
            return true;
        }
        const double incumbent = incumbent_.get();
        const double bound = global_live_bound(incumbent);
        if (fastbb::gap_within_tolerance(
                bound, incumbent, cfg_.abs_gap_tol, cfg_.rel_gap_tol)) {
            final_dual_bound_ = bound;
            reached_tolerance_ = true;
            signal_done();
            return true;
        }
        return false;
    }

    double node_tolerance(const Node& nd) const {
        return fastbb::depth_dependent_tolerance(
            cfg_.gpu_eps, cfg_.gpu_eps_decay, cfg_.gpu_eps_min, nd.depth);
    }

    bool should_prune(const Node& nd) const {
        return nd.dual_bound >= incumbent_.get();
    }

    void try_rounded_incumbent(
        const std::vector<double>& x,
        const Node&                nd,
        bool&                      new_incumbent)
    {
        auto candidate = make_integer_candidate(
            lmo_, x, nd.lb, nd.ub, prob_.int_vars);
        if (!is_rounded_candidate_feasible(
                lmo_, candidate, nd.lb, nd.ub, cfg_.integer_tol))
            return;
        new_incumbent = incumbent_.try_update(
            obj_.f(candidate), candidate) || new_incumbent;
    }

    void emit_progress(int iter, bool new_incumbent) {
        if (!progress_cb_) return;
        const double incumbent = incumbent_.get();
        const double bound = global_live_bound(incumbent);
        const int open_nodes =
            queue_.size() + active_tasks_.load(std::memory_order_acquire);
        progress_cb_(iter, open_nodes, bound, incumbent, new_incumbent);
    }
};
