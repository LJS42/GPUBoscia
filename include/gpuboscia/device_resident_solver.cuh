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
// on the device for iterate values, dual bounds, and the best LMO vertex.
// Each block also scores feasible integer candidates. value_local() returns the
// sum of the contributions assigned to the calling thread's block-stride
// range so it remains correct when n_vars exceeds blockDim.x.
//
// LMO requirements:
//   See lmo.cuh.  The LMO template argument is a public extension point:
//   users provide CUDA-copyable LMOs with compute_v() and a cooperative
//   check_feasible_device(). Nodes and integer candidates are checked on GPU.

#include <cuda_runtime.h>
#include "multiblock.cuh"
#include <cooperative_groups.h>
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
#include <unordered_set>
#include <vector>

#include "algorithms.cuh"
#include "atomic_incumbent.h"
#include "bnb.h"
#include "branching.h"
#include "concurrent_queue.h"
#include "gap_utils.h"
#include "heuristics.h"
#include "lmo.cuh"
#include "node.h"

namespace cg = cooperative_groups;
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

// Existing single-block reduction order remains unchanged by default.
template<class Objective>
__device__ double drs_gap_reduce(const Objective&, double* gradient,
    const double* x, const double* vertex, int n, int tid, int nw, double* ws) {
    double local = 0;
    for (int j = tid; j < n; j += blockDim.x)
        local += gradient[j] * (x[j] - vertex[j]);
    return drs_block_reduce(local, tid, nw, ws);
}

template<class Objective>
__device__ double drs_gap_reduce(const DRSMultiBlockObjective<Objective>& obj,
    double* gradient, const double* x, const double* vertex,
    int n, int tid, int nw, double* ws) {
    if (tid == 0) {
        obj.control->gradient = gradient;
        obj.control->x = x;
        obj.control->direction = vertex;
        obj.control->command = DRSNodeCommand::Gap;
    }
    obj.ctx.node_sync();
    double local = 0;
    for (int j = obj.ctx.global_tid; j < n; j += obj.ctx.total_threads)
        local += gradient[j] * (x[j] - vertex[j]);
    const double partial = drs_block_reduce(local, tid, nw, ws);
    if (tid == 0) obj.control->partials[0] = partial;
    obj.ctx.node_sync();
    if (tid == 0) {
        double gap = 0;
        for (int b = 0; b < obj.ctx.block_count; ++b) gap += obj.control->partials[b];
        obj.control->gap_result = gap;
    }
    __syncthreads();
    return obj.control->gap_result;
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

enum DRSNodeState : uint8_t {
    DRSFractionalUnconverged = 0,
    DRSFractionalConverged = 1,
    DRSInfeasible = 2,
    DRSIntegralUnconverged = 3,
    DRSIntegralConverged = 4
};

// Every oracle checks node feasibility cooperatively in the owning block.
// Return a block-uniform bool; vertex/direction are scratch vectors of size n.
template<typename LMO>
__device__ bool drs_node_feasible(const LMO& lmo, int tid, int n,
    const double* lb, const double* ub, const uint8_t* flags,
    double* vertex, double* direction) {
    bool valid = true;
    for (int j = tid; j < n; j += blockDim.x)
        if (isnan(lb[j]) || isnan(ub[j]) || lb[j] > ub[j]
                || (flags && flags[j] && ceil(lb[j]) > floor(ub[j])))
            valid = false;
    if (!__syncthreads_and(valid)) return false;
    return lmo.check_feasible_device(tid, n, lb, ub, flags, vertex, direction);
}

__device__ inline bool drs_near_integer(int tid, int n, const double* x,
    const uint8_t* flags, double tol) {
    bool integral = true;
    for (int j = tid; j < n; j += blockDim.x)
        if (!isfinite(x[j]) || (flags && flags[j]
                && fabs(x[j] - round(x[j])) > tol)) integral = false;
    return __syncthreads_and(integral);

}

// Custom LMOs may provide a cooperative, block-uniform hook
// candidate_feasible_device(tid, n, x, lb, ub, tol). The legacy signature
// without tid is called by thread zero. An optional integer_candidate_device
// (tid, n, x, lb, ub, integer_flags, candidate). Unknown sets skip rounding.
template<typename LMO, typename = void>
struct DRSHasCandidateFeasibility : std::false_type {};
template<typename LMO>
struct DRSHasCandidateFeasibility<LMO, std::void_t<decltype(
    std::declval<const LMO&>().candidate_feasible_device(0,
        (const double*)nullptr, (const double*)nullptr,
        (const double*)nullptr, 0.0))>> : std::true_type {};

template<typename LMO, typename = void>
struct DRSHasCooperativeCandidateFeasibility : std::false_type {};
template<typename LMO>
struct DRSHasCooperativeCandidateFeasibility<LMO, std::void_t<decltype(
    std::declval<const LMO&>().candidate_feasible_device(0, 0,
        (const double*)nullptr, (const double*)nullptr,
        (const double*)nullptr, 0.0))>> : std::true_type {};

template<typename LMO>
__device__ bool drs_rounded_feasible(const LMO& lmo, int tid, int n,
    const double* x, const double* lb, const double* ub, double tol) {
    if constexpr (DRSHasCooperativeCandidateFeasibility<LMO>::value) {
        return lmo.candidate_feasible_device(tid, n, x, lb, ub, tol);
    } else if constexpr (DRSHasCandidateFeasibility<LMO>::value) {
        __shared__ bool feasible;
        if (tid == 0) feasible = lmo.candidate_feasible_device(n, x, lb, ub, tol);
        __syncthreads();
        return feasible;
    }
    return false;
}
__device__ inline bool drs_rounded_feasible(const BoxLMO&, int, int,
    const double*, const double*, const double*, double) { return true; }
template<int MaxN, bool Equality>
__device__ bool drs_rounded_feasible(const UnitSimplexLMO<MaxN, Equality>& lmo,
    int tid, int n, const double* x, const double*, const double*, double tol) {
    double sum = 0.0;
    bool valid = true;
    for (int j = tid; j < n; j += blockDim.x) {
        valid &= x[j] >= -tol;
        sum += x[j];
    }
    const bool nonnegative = __syncthreads_and(valid);
    sum = feasibility_block_sum(sum);
    return nonnegative && (Equality ? fabs(sum - lmo.right_side) <= tol
                                   : sum <= lmo.right_side + tol);
}
template<int Order>
__device__ bool drs_rounded_feasible(const BirkhoffLMO<Order>&, int tid, int n,
    const double* x, const double*, const double*, double tol) {
    if (n != Order * Order) return false;
    bool valid = true;
    for (int i = tid; i < Order; i += blockDim.x) {
        double row = 0.0, col = 0.0;
        for (int j = 0; j < Order; ++j) {
            if (x[i * Order + j] < -tol) valid = false;
            row += x[i * Order + j];
            col += x[j * Order + i];
        }
        if (fabs(row - 1.0) > tol || fabs(col - 1.0) > tol) valid = false;
    }
    return __syncthreads_and(valid);
}

// LMO vertices already satisfy the oracle's set. Check exact integrality,
// finite coordinates and branch bounds before retaining their GPU score.
template<typename Objective>
__device__ double drs_integer_value(const Objective& obj, int tid, int n,
    int nw, const double* x, const double* lb, const double* ub,
    const uint8_t* flags, double tol, double* ws, unsigned char* obj_mem) {
    __shared__ double score;
    bool valid = true;
    for (int j = tid; j < n; j += blockDim.x)
        if (!isfinite(x[j]) || x[j] < lb[j] - tol || x[j] > ub[j] + tol
                || (flags && flags[j] && x[j] != round(x[j])))
            valid = false;
    if (!__syncthreads_and(valid)) return 1e300;
    const double value = drs_block_reduce(
        obj.value_local(tid, x, obj_mem), tid, nw, ws);
    if (tid == 0) score = isfinite(value) ? value : 1e300;
    __syncthreads();
    return score;
}

template<typename Objective, typename LMO, typename Heuristic>
__device__ void drs_apply_heuristic_and_score(const Objective& obj, const LMO& lmo,
    const Heuristic& heuristic, int tid, int n, int nw, const double* x, const double* lb, const double* ub,
    const uint8_t* flags, double tol, double* candidate, double* best,
    double* best_value, double* ws, unsigned char* obj_mem) {
    if (!flags) return;
    drs_generate_candidate(obj,lmo,heuristic,tid,n,x,lb,ub,flags,candidate);
    __syncthreads();
    __shared__ bool better;
    const bool feasible = drs_rounded_feasible(lmo, tid, n, candidate, lb, ub, tol);
    if (!feasible) return;
    const double value = drs_integer_value(
        obj, tid, n, nw, candidate, lb, ub, flags, tol, ws, obj_mem);
    if (tid == 0) better = value < *best_value;
    __syncthreads();
    if (better) {
        for (int j = tid; j < n; j += blockDim.x) best[j] = candidate[j];
        if (tid == 0) *best_value = value;
    }
    __syncthreads();
}

template<typename DeviceObjective, typename Algorithm, typename LMO,
         typename Heuristic = gpuboscia::RoundingHeuristic>
__device__
void drs_fw_relax_node(
    DeviceObjective obj,
    Algorithm       alg,
    LMO             lmo,
    double*         x,
    const double*   lb,
    const double*   ub,
    double*         out_gap,
    double*         out_value,
    int*            out_iterations,
    uint8_t*        converged,
    double*         best_vertex,
    double*         best_vertex_val,
    double*         workspace,
    unsigned char*  objective_workspace,
    size_t          objective_workspace_stride,
    int             N,
    int             max_iters,
    const double*   eps_by_node,
    const uint8_t* integer_flags, double candidate_tol,
    Heuristic heuristic, int node_index)
{
    constexpr int n = DeviceObjective::n_vars;
    static_assert(DeviceObjective::block_threads >= 32,
                  "block_threads must allow at least one CUDA warp");
    static_assert(DeviceObjective::block_threads % 32 == 0,
                  "block_threads must be a multiple of the CUDA warp size");
    const int node = node_index;
    const int tid  = threadIdx.x;
    const int nw   = blockDim.x >> 5;
    if (node >= N) return;
    const double eps = eps_by_node[node];
    const int initial_iterations = out_iterations[node];

    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int work_vectors =
        2 + (DRSUsesDICG<Algorithm>::value ? 2 : 0)
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
        out_iterations[node] = initial_iterations;
    }
    __syncthreads();

    if (!drs_node_feasible(lmo, tid, n, lbn, ubn, integer_flags, v_s, grad)) {
        if (tid == 0) {
            converged[node] = DRSInfeasible; // Infeasible, no objective evaluation.
            out_value[node] = CUDART_INF;
            out_gap[node] = 0.0;
        }
        return;
    }

    obj.prepare_block(tid, obj_smem);
    __syncthreads();

    if (initial_iterations == 0) {
        // Obtain a feasible starting vertex before asking the objective for a
        // gradient.  The bound vector lbn need not satisfy a custom LMO's set.
        for (int j = tid; j < n; j += blockDim.x)
            grad[j] = -1.0 - static_cast<double>(j) / (n + 1.0);
        __syncthreads();
        drs_compute_v(obj, lmo, tid, n, grad, lbn, ubn, v_s);
        __syncthreads();
        for (int j = tid; j < n; j += blockDim.x)
            xn[j] = v_s[j];
        __syncthreads();
    }

    for (int iter = 0; iter < max_iters; iter++) {
        if (tid == 0)
            out_iterations[node] = initial_iterations + iter + 1;
        __syncthreads();
        obj.grad(tid, xn, grad, obj_smem);
        __syncthreads();

        drs_compute_v(obj, lmo, tid, n, grad, lbn, ubn, v_s);
        __syncthreads();

        const double fv = drs_integer_value(obj, tid, n, nw, v_s, lbn, ubn,
                integer_flags, candidate_tol, ws, obj_smem);
        if (tid == 0 && fv < best_val_s) {
            best_val_s = fv;
            best_vertex_val[node] = fv;
        }
        if (tid == 0)
            fv_s = fv;
        __syncthreads();
        if (fv_s < 1e299 && fv_s == best_val_s)
            for (int j = tid; j < n; j += blockDim.x)
                best_vn[j] = v_s[j];
        __syncthreads();

        const double gap = drs_gap_reduce(obj, grad, xn, v_s, n, tid, nw, ws);

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

            drs_update_partition(obj, alg, n, xn, v_s, gamma, true);
        } else if constexpr (DRSUsesDICG<Algorithm>::value) {
            // The in-face routine minimizes its direction.  Negating the
            // objective gradient therefore returns the away maximizer a.
            for (int j = tid; j < n; j += blockDim.x)
                grad[j] = -grad[j];
            __syncthreads();
            drs_compute_inface(obj,lmo,tid,n,grad,xn,lbn,ubn,away_s);
            __syncthreads();

            // The in-face oracle also returns a feasible extreme point, so it
            // is an incumbent candidate just like the forward LMO vertex.
            const double away_f = drs_integer_value(obj, tid, n, nw, away_s, lbn, ubn,
                integer_flags, candidate_tol, ws, obj_smem);
            if (tid == 0 && away_f < best_val_s) {
                best_val_s = away_f;
                best_vertex_val[node] = away_f;
            }
            if (tid == 0)
                fv_s = away_f;
            __syncthreads();
            if (fv_s < 1e299 && fv_s == best_val_s)
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
                drs_update_partition(obj, alg, n, xn, away_s, gamma, true);
            } else {
                if (tid == 0) {
                    const double gamma_max = gamma_s;
                    const double open_loop_step =
                        alg.step_size(initial_iterations + iter, gap_s);
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
                drs_update_partition(obj, alg, n, xn, away_s, gamma_s, false);
            }
        } else {
            const double gamma = alg.step_size(initial_iterations + iter, gap_s);
            drs_update_partition(obj, alg, n, xn, v_s, gamma, false);
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

    drs_compute_v(obj, lmo, tid, n, grad, lbn, ubn, v_s);
    __syncthreads();

    const double final_fv = drs_integer_value(obj, tid, n, nw, v_s, lbn, ubn,
                integer_flags, candidate_tol, ws, obj_smem);
    if (tid == 0 && final_fv < best_val_s) {
        best_val_s = final_fv;
        best_vertex_val[node] = final_fv;
    }
    if (tid == 0)
        fv_s = final_fv;
    __syncthreads();
    if (fv_s < 1e299 && fv_s == best_val_s)
        for (int j = tid; j < n; j += blockDim.x)
            best_vn[j] = v_s[j];
    __syncthreads();

    const double final_gap = drs_gap_reduce(obj, grad, xn, v_s, n, tid, nw, ws);
    __syncthreads();
    const double final_value = drs_block_reduce(
        obj.value_local(tid, xn, obj_smem), tid, nw, ws);
    if (tid == 0) {
        out_value[node] = final_value;
        out_gap[node] = final_gap;
        converged[node] = alg.check_converged(final_gap, eps) ? 1 : 0;
    }
    __syncthreads();
    drs_apply_heuristic_and_score(obj, lmo, heuristic, tid, n, nw, xn, lbn, ubn,
        integer_flags, candidate_tol, v_s, best_vn,
        best_vertex_val + node, ws, obj_smem);
    if (drs_near_integer(tid, n, xn, integer_flags, candidate_tol) && tid == 0)
        converged[node] = converged[node]
            ? DRSIntegralConverged : DRSIntegralUnconverged;
}

template<typename DeviceObjective, typename Algorithm, typename LMO,
         typename Heuristic = gpuboscia::RoundingHeuristic>
__global__
void drs_fw_relax_batch_kernel(
    DeviceObjective obj,
    Algorithm       alg,
    LMO             lmo,
    double*         x,
    const double*   lb,
    const double*   ub,
    double*         out_gap,
    double*         out_value,
    int*            out_iterations,
    uint8_t*        converged,
    double*         best_vertex,
    double*         best_vertex_val,
    double*         workspace,
    unsigned char*  objective_workspace,
    size_t          objective_workspace_stride,
    int             N,
    int             max_iters,
    const double*   eps_by_node,
    const uint8_t* integer_flags = nullptr, double candidate_tol = 1e-9,
    Heuristic heuristic = {})
{
    drs_fw_relax_node(obj, alg, lmo, x, lb, ub, out_gap, out_value,
        out_iterations, converged, best_vertex, best_vertex_val, workspace,
        objective_workspace, objective_workspace_stride, N, max_iters,
        eps_by_node, integer_flags, candidate_tol, heuristic, blockIdx.x);
}

template<typename DeviceObjective, typename Algorithm, typename LMO,
         typename Heuristic = gpuboscia::RoundingHeuristic>
__global__
void drs_fw_multiblock_kernel(
    DeviceObjective obj,
    Algorithm       alg,
    LMO             lmo,
    double*         x,
    const double*   lb,
    const double*   ub,
    double*         out_gap,
    double*         out_value,
    int*            out_iterations,
    uint8_t*        converged,
    double*         best_vertex,
    double*         best_vertex_val,
    double*         workspace,
    unsigned char*  objective_workspace,
    size_t          objective_workspace_stride,
    int             N,
    int             max_iters,
    const double*   eps_by_node,
    const uint8_t* integer_flags, double candidate_tol,
    Heuristic heuristic, int blocks_per_node, DRSNodeControl* controls)
{
    if constexpr (DRS_HAS_MULTIBLOCK_ATOMICS && DRSHasMultiBlockGradient<DeviceObjective>::value) {
        const int node = blockIdx.x / blocks_per_node;
        auto& control = controls[node];
        MultiBlockContext ctx(node, blockIdx.x % blocks_per_node,
                              blocks_per_node, &control.barrier);
        ctx.lmo_workspace = reinterpret_cast<unsigned char*>(controls)
            + drs_multiblock_scratch_offset(N, blocks_per_node)
            + size_t(node) * DRSLMOWorkspace<LMO>::bytes;
        ctx.reduction_partials = reinterpret_cast<double*>(controls+N)
            + static_cast<size_t>(node)*blocks_per_node;
        extern __shared__ __align__(16) unsigned char reduction_smem[];
        double* reduction_ws = reinterpret_cast<double*>(reduction_smem);
        DRSMultiBlockObjective<DeviceObjective> collective{obj, ctx, &control};
        if (ctx.block_rank == 0) {
            // Packed per-block scratch follows the N control records. No
            // node vectors are replicated as K grows.
            if (threadIdx.x == 0)
                control.partials = reinterpret_cast<double*>(controls + N)
                    + static_cast<size_t>(node) * blocks_per_node;
            drs_fw_relax_node(collective, alg, lmo, x, lb, ub, out_gap,
                out_value, out_iterations, converged, best_vertex,
                best_vertex_val, workspace, objective_workspace,
                objective_workspace_stride, N, max_iters, eps_by_node,
                integer_flags, candidate_tol, heuristic, node);
            if (threadIdx.x == 0) control.command = DRSNodeCommand::Stop;
            ctx.node_sync();
        } else {
            for (;;) {
                ctx.node_sync();
                if (control.command == DRSNodeCommand::Stop) break;
                if (control.command == DRSNodeCommand::Gradient) {
                    obj.grad_multiblock(ctx, control.x, control.gradient,
                                        control.objective_workspace);
                } else if (control.command == DRSNodeCommand::ForwardLMO) {
                    if constexpr (DRSHasMultiBlockLMO<LMO>::value)
                        lmo.compute_v_multiblock(ctx, control.lmo_n, control.x,
                            control.lmo_lb, control.lmo_ub, control.update_x);
                } else if (control.command == DRSNodeCommand::Value) {
                    if constexpr (DRSHasMultiBlockValue<DeviceObjective>::value)
                        obj.value_multiblock(ctx,control.x,control.objective_workspace);
                } else if (control.command == DRSNodeCommand::FactorValue) {
                    if constexpr (DRSHasMultiBlockFactorValue<DeviceObjective>::value)
                        obj.value_from_factor_multiblock(ctx,control.objective_workspace);
                } else if (control.command == DRSNodeCommand::Curvature) {
                    if constexpr (DRSHasMultiBlockCurvature<DeviceObjective>::value)
                        obj.quadratic_curvature_multiblock(ctx,control.direction,control.objective_workspace);
                } else if (control.command == DRSNodeCommand::InFaceLMO) {
                    if constexpr (DRSHasMultiBlockInFace<LMO>::value)
                        lmo.compute_inface_v_multiblock(ctx,control.lmo_n,control.x,control.face_x,control.lmo_lb,control.lmo_ub,control.update_x);
                } else if (control.command == DRSNodeCommand::Heuristic) {
                    if constexpr (DRSHasMultiBlockHeuristic<Heuristic,LMO>::value)
                        heuristic.generate_multiblock(ctx,lmo,control.lmo_n,control.x,control.lmo_lb,control.lmo_ub,control.integer_flags,control.update_x);
                } else if (control.command == DRSNodeCommand::Gap) {
                    double local = 0;
                    for (int j = ctx.global_tid; j < DeviceObjective::n_vars; j += ctx.total_threads)
                        local += control.gradient[j] * (control.x[j] - control.direction[j]);
                    const double partial = drs_block_reduce(local, threadIdx.x, blockDim.x >> 5, reduction_ws);
                    if (threadIdx.x == 0) control.partials[ctx.block_rank] = partial;
                } else {
                    for (int j = ctx.global_tid; j < DeviceObjective::n_vars;
                         j += ctx.total_threads) {
                        if (control.command == DRSNodeCommand::AdditiveUpdate)
                            control.update_x[j] += control.gamma * control.direction[j];
                        else alg.update(control.update_x, control.direction, control.gamma, j);
                    }
                }
                ctx.node_sync();
            }
        }
    }
}

// Low-frontier DICG kernel.  A cooperative grid assigns two complete blocks
// to every node.  The even block computes the forward oracle and owns the FW
// state; the odd block computes the in-face oracle at the same time.  Grid
// barriers are safe here because the launcher admits the kernel only when its
// complete grid is resident.
template<typename DeviceObjective, typename Algorithm, typename LMO,
         typename Heuristic = gpuboscia::RoundingHeuristic>
__global__ void drs_dicg_paired_batch_kernel(
    DeviceObjective obj, Algorithm alg, LMO lmo,
    double* x, const double* lb, const double* ub,
    double* out_gap, double* out_value, int* out_iterations, uint8_t* converged,
    double* best_vertex, double* best_vertex_val, double* workspace,
    unsigned char* objective_workspace, size_t objective_workspace_stride,
    int N, int max_iters, const double* eps_by_node, int* all_done,
    const uint8_t* integer_flags = nullptr, double candidate_tol = 1e-9,
    Heuristic heuristic = {})
{
    static_assert(DRSUsesDICG<Algorithm>::value,
                  "paired oracle kernel is DICG-only");
    constexpr int n = DeviceObjective::n_vars;
    cg::grid_group grid = cg::this_grid();
    const int node = (int)blockIdx.x >> 1;
    const bool forward = ((int)blockIdx.x & 1) == 0;
    const int initial_iterations = out_iterations[node];
    const int tid = threadIdx.x;
    const int nw = blockDim.x >> 5;
    if (node >= N) return;

    // Four vectors per node: the ordinary kernel uses the first three.  The
    // fourth holds -grad so the in-face LMO never aliases the forward block's
    // gradient while both oracles execute.
    constexpr int paired_vectors =
        4 + (Algorithm::uses_secant_line_search ? 2 : 0);
    double* base = workspace + (size_t)node * paired_vectors * n;
    double* grad = base;
    double* v = grad + n;
    double* away = v + n;
    double* neg_grad = away + n;
    double* trial_x = nullptr;
    double* trial_grad = nullptr;
    if constexpr (Algorithm::uses_secant_line_search) {
        trial_x = neg_grad + n;
        trial_grad = trial_x + n;
    }
    double* xn = x + (size_t)node * n;
    const double* lbn = lb + (size_t)node * n;
    const double* ubn = ub + (size_t)node * n;
    double* best = best_vertex + (size_t)node * n;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    double* ws = reinterpret_cast<double*>(smem_raw);
    unsigned char* obj_mem = objective_workspace
        + (size_t)node * objective_workspace_stride;
    obj_mem = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<uintptr_t>(obj_mem) + 15u) & ~uintptr_t(15u));

    __shared__ double gap_s, gamma_s, candidate_s;
    __shared__ double phi_lo_s, phi_hi_s, lo_s, hi_s;
    __shared__ double f_base_s, best_f_s, best_gamma_s;
    __shared__ bool done_s, line_done_s, candidate_better_s;
    if (forward) {
        if (tid == 0) {
            converged[node] = 0;
            out_iterations[node] = initial_iterations;
            best_vertex_val[node] = 1e300;
        }
        const bool feasible = drs_node_feasible(
            lmo, tid, n, lbn, ubn, integer_flags, v, grad);
        if (!feasible && tid == 0) {
            converged[node] = DRSInfeasible;
            out_value[node] = CUDART_INF;
            out_gap[node] = 0.0;
        }
        __syncthreads();
        if (feasible) {
            obj.prepare_block(tid, obj_mem);
            __syncthreads();
            if (initial_iterations == 0) {
                for (int j = tid; j < n; j += blockDim.x)
                    grad[j] = -1.0 - static_cast<double>(j) / (n + 1.0);
                __syncthreads();
                lmo.compute_v(tid, n, grad, lbn, ubn, v);
                __syncthreads();
                for (int j = tid; j < n; j += blockDim.x) xn[j] = v[j];
            }
        }
    }
    grid.sync();

    for (int iter = 0; iter < max_iters; ++iter) {
        if (forward && converged[node] == 0) {
            if (tid == 0) out_iterations[node] = initial_iterations + iter + 1;
            obj.grad(tid, xn, grad, obj_mem);
        }
        grid.sync();

        if (forward && converged[node] == 0) {
            lmo.compute_v(tid, n, grad, lbn, ubn, v);
        } else if (!forward && converged[node] == 0) {
            for (int j = tid; j < n; j += blockDim.x)
                neg_grad[j] = -grad[j];
            __syncthreads();
            lmo.compute_inface_v(tid, n, neg_grad, xn, lbn, ubn, away);
        }
        grid.sync();

        if (forward && converged[node] == 0) {
            const double fv = drs_integer_value(obj, tid, n, nw, v, lbn, ubn,
                integer_flags, candidate_tol, ws, obj_mem);
            if (tid == 0) {
                candidate_s = fv;
                candidate_better_s = fv < best_vertex_val[node];
            }
            __syncthreads();
            if (candidate_better_s) {
                for (int j = tid; j < n; j += blockDim.x) best[j] = v[j];
                if (tid == 0) best_vertex_val[node] = candidate_s;
            }
            __syncthreads();
            double local_gap = 0.0;
            for (int j = tid; j < n; j += blockDim.x)
                local_gap += grad[j] * (xn[j] - v[j]);
            const double gap = drs_block_reduce(local_gap, tid, nw, ws);
            if (tid == 0) {
                gap_s = gap;
                done_s = alg.check_converged(gap, eps_by_node[node]);
                out_gap[node] = gap;
                converged[node] = done_s ? 1 : 0;
            }
            __syncthreads();
            if (!done_s) {
                const double af = drs_integer_value(obj, tid, n, nw, away, lbn, ubn,
                integer_flags, candidate_tol, ws, obj_mem);
                if (tid == 0) {
                    candidate_s = af;
                    candidate_better_s = af < best_vertex_val[node];
                }
                __syncthreads();
                if (candidate_better_s) {
                    for (int j = tid; j < n; j += blockDim.x)
                        best[j] = away[j];
                    if (tid == 0) best_vertex_val[node] = candidate_s;
                }
                __syncthreads();
                for (int j = tid; j < n; j += blockDim.x)
                    away[j] -= v[j];
                __syncthreads();
                const double coordinate = drs_block_min(
                    lmo.dicg_maximum_step_local(
                        tid, n, away, xn, lbn, ubn), tid, nw, ws);
                const double aggregate =
                    lmo.dicg_sum_maximum_step(tid, n, away, xn);
                if (tid == 0)
                    gamma_s = fmax(0.0, fmin(coordinate, aggregate));
                __syncthreads();
                if constexpr (Algorithm::uses_secant_line_search) {
                    for (int j = tid; j < n; j += blockDim.x)
                        away[j] = -away[j];
                    __syncthreads();
                    double local_pairwise_gap = 0.0;
                    for (int j = tid; j < n; j += blockDim.x)
                        local_pairwise_gap -= grad[j] * away[j];
                    const double pairwise_gap = drs_block_reduce(
                        local_pairwise_gap, tid, nw, ws);
                    const double gamma = alg.ls.search(
                        obj, tid, n, nw, pairwise_gap, gamma_s, xn, away,
                        trial_x, trial_grad, ws, obj_mem,
                        &gamma_s, &phi_lo_s, &phi_hi_s, &lo_s, &hi_s,
                        &f_base_s, &best_f_s, &best_gamma_s, &line_done_s);
                    for (int j = tid; j < n; j += blockDim.x)
                        xn[j] += gamma * away[j];
                } else {
                    if (tid == 0) {
                        const double maximum = gamma_s;
                        const double scheduled = alg.step_size(initial_iterations + iter, gap_s);
                        if constexpr (DRSSafeguardsDICGDropStep<Algorithm>::value)
                            gamma_s = maximum <= scheduled
                                ? maximum : maximum * scheduled;
                        else if constexpr (DRSScalesDICGMaximumStep<Algorithm>::value)
                            gamma_s = maximum * scheduled;
                        else
                            gamma_s = fmin(maximum, scheduled);
                    }
                    __syncthreads();
                    for (int j = tid; j < n; j += blockDim.x)
                        alg.update(xn, away, gamma_s, j);
                }
            }
        }
        grid.sync();

        if (blockIdx.x == 0 && tid == 0) {
            int done = 1;
            for (int i = 0; i < N; ++i) done &= converged[i] != 0;
            *all_done = done;
        }
        grid.sync();
        if (*all_done) break;
    }

    if (!forward || converged[node] == DRSInfeasible) return;
    // Return a certificate coherent with the final iterate, matching the
    // ordinary kernel's post-loop behavior.
    obj.grad(tid, xn, grad, obj_mem);
    __syncthreads();
    lmo.compute_v(tid, n, grad, lbn, ubn, v);
    __syncthreads();
    const double final_fv = drs_integer_value(obj, tid, n, nw, v, lbn, ubn,
                integer_flags, candidate_tol, ws, obj_mem);
    if (tid == 0) {
        candidate_s = final_fv;
        candidate_better_s = final_fv < best_vertex_val[node];
    }
    __syncthreads();
    if (candidate_better_s) {
        for (int j = tid; j < n; j += blockDim.x) best[j] = v[j];
        if (tid == 0) best_vertex_val[node] = candidate_s;
    }
    __syncthreads();
    double local_gap = 0.0;
    for (int j = tid; j < n; j += blockDim.x)
        local_gap += grad[j] * (xn[j] - v[j]);
    const double final_gap = drs_block_reduce(local_gap, tid, nw, ws);
    __syncthreads();
    const double final_value = drs_block_reduce(
        obj.value_local(tid, xn, obj_mem), tid, nw, ws);
    if (tid == 0) {
        out_value[node] = final_value;
        out_gap[node] = final_gap;
        converged[node] = alg.check_converged(
            final_gap, eps_by_node[node]) ? 1 : 0;
    }
    __syncthreads();
    drs_apply_heuristic_and_score(obj, lmo, heuristic, tid, n, nw, xn, lbn, ubn,
        integer_flags, candidate_tol, v, best,
        best_vertex_val + node, ws, obj_mem);
    if (drs_near_integer(tid, n, xn, integer_flags, candidate_tol) && tid == 0)
        converged[node] = converged[node]
            ? DRSIntegralConverged : DRSIntegralUnconverged;
}

enum class FWExecutionMode { SingleBlock, MultiBlock };

struct DeviceResidentConfig {
    enum class DICGOracleParallelism { Auto, Serial, Paired };
    int    gpu_batch_size = 1024;
    int    num_streams    = 8;
    // 0 selects a warp-aligned size automatically. A positive override must
    // be a multiple of the CUDA warp size (32).
    int    gpu_block_threads = 0;
    int    gpu_max_iters  = 20000;
    double gpu_eps        = 1e-3;
    double gpu_eps_decay  = 0.8;
    double gpu_eps_min    = 1e-6;
    double integer_tol    = 1e-6;
    double abs_gap_tol    = 1e-6;
    double rel_gap_tol    = 1e-2;
    DICGOracleParallelism dicg_oracle_parallelism =
        DICGOracleParallelism::Auto;
    // Fraction of the cooperative-grid residency limit usable by Auto.
    double dicg_paired_occupancy_fraction = 0.8;
    // Appended to preserve positional aggregate initialization of older fields.
    FWExecutionMode execution_mode = FWExecutionMode::SingleBlock;
    int blocks_per_node = 4;
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
    bool         paired_dicg = false;
    std::uint64_t generation = 0;

    DRSCompletionSignal sig{};
    std::vector<Node>   batch_nodes;
    std::thread         slot_thread;

    double*  h_x   = nullptr;
    double*  h_lb  = nullptr;
    double*  h_ub  = nullptr;
    double*  h_gap = nullptr;
    double*  h_value = nullptr;
    double*  h_eps = nullptr;
    int*     h_iterations = nullptr;
    double*  h_best_vertex = nullptr;
    double*  h_best_vertex_val = nullptr;
    uint8_t* h_cnv = nullptr;
    uint8_t* h_integer_flags = nullptr;

    double*  d_x   = nullptr;
    double*  d_lb  = nullptr;
    double*  d_ub  = nullptr;
    double*  d_gap = nullptr;
    double*  d_value = nullptr;
    double*  d_eps = nullptr;
    int*     d_iterations = nullptr;
    double*  d_best_vertex = nullptr;
    double*  d_best_vertex_val = nullptr;
    double*  d_workspace = nullptr;
    unsigned char* d_objective_workspace = nullptr;
    uint8_t* d_cnv = nullptr;
    uint8_t* d_integer_flags = nullptr;
    int*     d_all_done = nullptr;
    DRSNodeControl* d_node_controls = nullptr;

    void alloc(int cap, int n, int work_vectors, size_t objective_stride) {
        capacity = cap;
        const size_t xn = (size_t)cap * n;

        DRS_CUDA_CHECK(cudaHostAlloc(&h_x,   xn  * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_lb,  xn  * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_ub,  xn  * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_gap, cap * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_value, cap * sizeof(double), cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_eps, cap * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_iterations, cap * sizeof(int), cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_best_vertex,     xn  * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_best_vertex_val, cap * sizeof(double),  cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaHostAlloc(&h_cnv, cap * sizeof(uint8_t), cudaHostAllocDefault));

        DRS_CUDA_CHECK(cudaMalloc(&d_x,   xn  * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_lb,  xn  * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_ub,  xn  * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_gap, cap * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_value, cap * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_eps, cap * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_iterations, cap * sizeof(int)));
        DRS_CUDA_CHECK(cudaMalloc(&d_best_vertex,     xn  * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_best_vertex_val, cap * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&d_cnv, cap * sizeof(uint8_t)));
        DRS_CUDA_CHECK(cudaMalloc(&d_all_done, sizeof(int)));
        DRS_CUDA_CHECK(cudaMalloc(
            &d_workspace, xn * (size_t)work_vectors * sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(
            &d_objective_workspace, (size_t)cap * objective_stride));

        DRS_CUDA_CHECK(cudaHostAlloc(&h_integer_flags, n * sizeof(uint8_t), cudaHostAllocDefault));
        DRS_CUDA_CHECK(cudaMalloc(&d_integer_flags, n * sizeof(uint8_t)));
        DRS_CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void free_all() {
        if (slot_thread.joinable()) slot_thread.join();
        cudaFreeHost(h_x);
        cudaFreeHost(h_lb);
        cudaFreeHost(h_ub);
        cudaFreeHost(h_gap);
        cudaFreeHost(h_value);
        cudaFreeHost(h_eps);
        cudaFreeHost(h_iterations);
        cudaFreeHost(h_best_vertex);
        cudaFreeHost(h_best_vertex_val);
        cudaFreeHost(h_cnv);
        cudaFreeHost(h_integer_flags);
        cudaFree(d_x);
        cudaFree(d_lb);
        cudaFree(d_ub);
        cudaFree(d_gap);
        cudaFree(d_value);
        cudaFree(d_eps);
        cudaFree(d_iterations);
        cudaFree(d_best_vertex);
        cudaFree(d_best_vertex_val);
        cudaFree(d_cnv);
        cudaFree(d_integer_flags);
        cudaFree(d_all_done);
        cudaFree(d_node_controls);
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
    std::vector<uint8_t> branched_early;
    std::vector<uint8_t> node_state;
    std::vector<double> gap, value, lower_bound;
};

template<typename DeviceObjective, typename Algorithm, typename LMO = BoxLMO,
         typename Heuristic = gpuboscia::RoundingHeuristic>
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
            std::numeric_limits<double>::infinity(),
        Heuristic heuristic = {})
        : prob_(prob), obj_(obj), alg_(alg), lmo_(lmo), heuristic_(heuristic)
        , branch_sel_(std::move(branch_sel)), cfg_(cfg)
        , time_limit_seconds_(time_limit_seconds)
    {
        if (cfg_.gpu_max_iters <= 0 || cfg_.gpu_batch_size <= 0
                || cfg_.num_streams <= 0)
            throw std::invalid_argument("GPU iterations, batch size and streams must be positive");
        for (int j : prob_.int_vars)
            if (j < 0 || j >= prob_.n_vars)
                throw std::invalid_argument("integer variable index out of range");
        if (!std::isfinite(cfg_.gpu_eps) || cfg_.gpu_eps <= 0.0
                || !std::isfinite(cfg_.gpu_eps_min)
                || cfg_.gpu_eps_min <= 0.0
                || !std::isfinite(cfg_.gpu_eps_decay)
                || cfg_.gpu_eps_decay <= 0.0 || cfg_.gpu_eps_decay > 1.0
                || !std::isfinite(cfg_.abs_gap_tol) || cfg_.abs_gap_tol < 0.0
                || !std::isfinite(cfg_.rel_gap_tol) || cfg_.rel_gap_tol < 0.0
                || !std::isfinite(cfg_.dicg_paired_occupancy_fraction)
                || cfg_.dicg_paired_occupancy_fraction <= 0.0
                || cfg_.dicg_paired_occupancy_fraction > 1.0)
            throw std::invalid_argument("invalid tolerance configuration");
        if (std::isnan(time_limit_seconds_) || time_limit_seconds_ <= 0.0)
            throw std::invalid_argument("time limit must be positive");
        if (prob_.n_vars != DeviceObjective::n_vars)
            throw std::runtime_error("DeviceObjective::n_vars does not match Problem::n_vars");
        configure_kernel_shared_memory();
        configure_multiblock();
        slots_.resize(cfg_.num_streams);
        for (int i = 0; i < cfg_.num_streams; i++) {
            slots_[i].id = i;
            slots_[i].alloc(
                cfg_.gpu_batch_size, prob_.n_vars,
                kernel_work_vectors(), objective_workspace_stride());
            if (cfg_.execution_mode == FWExecutionMode::MultiBlock)
                DRS_CUDA_CHECK(cudaMalloc(&slots_[i].d_node_controls,
                    drs_multiblock_control_bytes(cfg_.gpu_batch_size, cfg_.blocks_per_node, DRSLMOWorkspace<LMO>::bytes)));
            std::fill(slots_[i].h_integer_flags,
                      slots_[i].h_integer_flags + prob_.n_vars, uint8_t(0));
            for (int j : prob_.int_vars) slots_[i].h_integer_flags[j] = 1;
            DRS_CUDA_CHECK(cudaMemcpyAsync(
                slots_[i].d_integer_flags, slots_[i].h_integer_flags,
                prob_.n_vars * sizeof(uint8_t), cudaMemcpyHostToDevice,
                slots_[i].stream));
            slots_[i].sig = { &completed_mtx_, &completed_cv_, &completed_, i };
            configure_l2_window(slots_[i].stream);
        }
    }

    ~DeviceResidentHybridSolver() {
        for (auto& slot : slots_)
            slot.free_all();
    }

    void push(const Node& nd) {
        if (nd.lb.size() != (size_t)prob_.n_vars || nd.ub.size() != nd.lb.size()
                || (!nd.relaxation_x.empty() && nd.relaxation_x.size() != nd.lb.size())
                || (nd.relaxation_iterations > 0 && nd.relaxation_x.empty())
                || nd.relaxation_iterations < 0)
            throw std::invalid_argument("invalid node buffer dimensions or resume state");
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
    int                 effective_gpu_batch_size() const { return cfg_.gpu_batch_size; }
    int                 effective_num_streams() const { return cfg_.num_streams; }
    int                 paired_gpu_batches() const {
        return paired_gpu_batches_.load(std::memory_order_relaxed);
    }

private:
    const Problem&       prob_;
    DeviceObjective      obj_;
    Algorithm            alg_;
    LMO                  lmo_;
    Heuristic      heuristic_;
    BranchStrategy       branch_sel_;
    DeviceResidentConfig cfg_;

    ConcurrentQueue queue_;
    ConcurrentBoundSet active_bounds_;
    // Only the host worker accesses this set. GPU work is not cancelled;
    // completed children of these parents are retired without expansion.
    std::unordered_set<int> pruned_early_parents_;
    AtomicIncumbent incumbent_;
    std::atomic<int> active_tasks_{0};
    std::atomic<int> next_id_{1};
    std::atomic<int> nodes_done_{0};
    std::atomic<int> last_gpu_iterations_{0};
    std::atomic<double> last_gpu_gap_{std::numeric_limits<double>::infinity()};
    std::atomic<bool> last_gpu_converged_{false};
    std::atomic<bool> done_{false};
    std::atomic<int> paired_gpu_batches_{0};
    double final_dual_bound_ = -std::numeric_limits<double>::infinity();
    bool reached_tolerance_ = false;
    bool reached_time_limit_ = false;
    using Clock = std::chrono::steady_clock;
    Clock::time_point solve_started_{};
    double time_limit_seconds_ = std::numeric_limits<double>::infinity();
    std::function<void(int, int, double, double, bool)> progress_cb_;
    std::function<void(const DRSSchedulingEvent&)> scheduling_cb_;
    mutable bool paired_kernel_resource_compatible_ = true;
    int multiblock_device_ = -1;

    std::vector<DRSStreamSlot> slots_;
    std::mutex                 completed_mtx_;
    std::condition_variable    completed_cv_;
    std::queue<int>            completed_;

    static constexpr int kernel_work_vectors() {
        return
            2 + (DRSUsesDICG<Algorithm>::value ? 2 : 0)
              + (Algorithm::uses_secant_line_search ? 2 : 0);
    }

    size_t objective_workspace_stride() const {
        static_assert(DeviceObjective::shared_bytes >= 0, "negative objective workspace");
        size_t bytes = std::max(size_t(16), size_t(DeviceObjective::shared_bytes));
        if constexpr (DRSHasMultiBlockWorkspace<DeviceObjective>::value) {
            if (cfg_.execution_mode == FWExecutionMode::MultiBlock)
                bytes = std::max(bytes, static_cast<size_t>(obj_.multiblock_workspace_bytes(
                    cfg_.blocks_per_node, kernel_block_threads(prob_.n_vars))));
        }
        if (bytes > std::numeric_limits<size_t>::max() - 15u)
            throw std::invalid_argument("MultiBlock objective workspace size overflows");
        const size_t stride = (bytes + 15u) & ~size_t(15u);
        if (stride > std::numeric_limits<size_t>::max() / size_t(cfg_.gpu_batch_size))
            throw std::invalid_argument("objective batch workspace size overflows");
        return stride;
    }

    static constexpr int kernel_dynamic_shared_bytes() {
        constexpr int nw = DeviceObjective::block_threads / 32;
        return nw * (int)sizeof(double);
    }

    int kernel_block_threads(int n) const {
        cudaFuncAttributes attrs{};
        if (cfg_.execution_mode == FWExecutionMode::MultiBlock) {
            DRS_CUDA_CHECK(cudaFuncGetAttributes(
                &attrs, drs_fw_multiblock_kernel<DeviceObjective, Algorithm, LMO, Heuristic>));
        } else {
            DRS_CUDA_CHECK(cudaFuncGetAttributes(
                &attrs, drs_fw_relax_batch_kernel<DeviceObjective, Algorithm, LMO, Heuristic>));
        }
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
        const long long wanted = ((static_cast<long long>(n) + 31) / 32) * 32;
        return static_cast<int>(std::min(std::max(32LL, wanted),
                                         static_cast<long long>(warp_limit)));
    }

    void configure_multiblock() {
        if (cfg_.execution_mode != FWExecutionMode::SingleBlock &&
            cfg_.execution_mode != FWExecutionMode::MultiBlock)
            throw std::invalid_argument("unknown FW execution mode");
        if (cfg_.execution_mode != FWExecutionMode::MultiBlock) return;
        if (cfg_.blocks_per_node < 1)
            throw std::invalid_argument("MultiBlock blocks_per_node must be positive");
        if constexpr (!DRS_HAS_MULTIBLOCK_ATOMICS) {
            throw std::runtime_error("MultiBlock requires CUDA Toolkit 11.6 or newer");
        } else if constexpr (!DRSHasMultiBlockGradient<DeviceObjective>::value) {
            throw std::invalid_argument("MultiBlock requires Objective::grad_multiblock");
        } else {
            if (cfg_.dicg_oracle_parallelism == DeviceResidentConfig::DICGOracleParallelism::Paired)
                throw std::invalid_argument("MultiBlock cannot combine with paired DICG");
            int device, cooperative, sms, active, max_grid_x, default_smem, optin_smem;
            DRS_CUDA_CHECK(cudaGetDevice(&device));
            multiblock_device_ = device;
            DRS_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
            if (!cooperative) throw std::runtime_error("MultiBlock requires cooperative launch support");
            DRS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            auto kernel = drs_fw_multiblock_kernel<DeviceObjective, Algorithm, LMO, Heuristic>;
            cudaFuncAttributes attrs{};
            DRS_CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
            DRS_CUDA_CHECK(cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, device));
            DRS_CUDA_CHECK(cudaDeviceGetAttribute(&default_smem, cudaDevAttrMaxSharedMemoryPerBlock, device));
            DRS_CUDA_CHECK(cudaDeviceGetAttribute(&optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
            const size_t total_smem = attrs.sharedSizeBytes + kernel_dynamic_shared_bytes();
            if (total_smem > static_cast<size_t>(optin_smem))
                throw std::invalid_argument("MultiBlock shared memory exceeds device limit");
            if (total_smem > static_cast<size_t>(default_smem))
                DRS_CUDA_CHECK(cudaFuncSetAttribute(kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_dynamic_shared_bytes()));
            const int threads = kernel_block_threads(prob_.n_vars);
            if (threads > attrs.maxThreadsPerBlock)
                throw std::invalid_argument("MultiBlock block size exceeds kernel limit");
            DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, kernel, threads, kernel_dynamic_shared_bytes()));
            cfg_.gpu_batch_size = drs_multiblock_batch_limit(cfg_.gpu_batch_size,
                cfg_.blocks_per_node, cooperative != 0, active, sms, max_grid_x);
            cfg_.num_streams = 1;
            // Validate optional Objective scratch before any slot allocations.
            (void)objective_workspace_stride();
        }
    }

    bool paired_dicg_available(int N, int n) const {
        if constexpr (!DRSUsesDICG<Algorithm>::value) {
            return false;
        } else {
            if (!paired_kernel_resource_compatible_) return false;
            int device = 0;
            cudaDeviceProp prop{};
            DRS_CUDA_CHECK(cudaGetDevice(&device));
            DRS_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            if (!prop.cooperativeLaunch) return false;
            const int threads = kernel_block_threads(n);
            int blocks_per_sm = 0;
            DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm,
                drs_dicg_paired_batch_kernel<DeviceObjective, Algorithm, LMO, Heuristic>,
                threads, kernel_dynamic_shared_bytes()));
            const int resident = blocks_per_sm * prop.multiProcessorCount;
            const bool forced = cfg_.dicg_oracle_parallelism ==
                DeviceResidentConfig::DICGOracleParallelism::Paired;
            const int allowed = (int)std::floor(
                (forced ? 1.0 : cfg_.dicg_paired_occupancy_fraction)
                * resident);
            return 2 * N <= allowed;
        }
    }

    bool choose_paired_dicg(int N, int n) const {
        if (cfg_.execution_mode == FWExecutionMode::MultiBlock) return false;
        using Mode = DeviceResidentConfig::DICGOracleParallelism;
        if (cfg_.dicg_oracle_parallelism == Mode::Serial) return false;
        bool other_active = false;
        for (const auto& active : slots_)
            if (active.in_use) { other_active = true; break; }
        const bool available = !other_active && paired_dicg_available(N, n);
        if (cfg_.dicg_oracle_parallelism == Mode::Paired && !available)
            throw std::runtime_error(
                "forced paired DICG requires an idle cooperative-capable GPU "
                "and a fully resident 2*N-block grid");
        return available;
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
            drs_fw_relax_batch_kernel<DeviceObjective, Algorithm, LMO, Heuristic>));

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
                drs_fw_relax_batch_kernel<DeviceObjective, Algorithm, LMO, Heuristic>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                dynamic_bytes));
        }
        if constexpr (DRSUsesDICG<Algorithm>::value) {
            cudaFuncAttributes paired_attrs{};
            DRS_CUDA_CHECK(cudaFuncGetAttributes(
                &paired_attrs,
                drs_dicg_paired_batch_kernel<
                    DeviceObjective, Algorithm, LMO, Heuristic>));
            const size_t paired_total =
                (size_t)paired_attrs.sharedSizeBytes + (size_t)dynamic_bytes;
            if (paired_total > (size_t)optin_limit) {
                paired_kernel_resource_compatible_ = false;
            } else if (paired_total > (size_t)default_limit) {
                DRS_CUDA_CHECK(cudaFuncSetAttribute(
                    drs_dicg_paired_batch_kernel<
                        DeviceObjective, Algorithm, LMO, Heuristic>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    dynamic_bytes));
            }
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

            if (queue_.empty() && has_dispatchable_slot())
                branch_starved_batches(completed_batches);

            for (auto& batch : completed_batches)
                process_completed_batch(batch);

            if (active_tasks_.load(std::memory_order_acquire) == 0
                    && finish_if_search_complete())
                return;

            // Processing a completed batch may have generated enough child
            // nodes to fill slots that could not be refilled above.
            fill_free_slots();

            bool gpu_active = false;
            for (auto& slot : slots_) {
                if (slot.in_use) {
                    gpu_active = true;
                }
            }

            // Never leave the GPU completely idle just because the frontier
            // is smaller than a full batch. This bootstraps the search (and
            // drains its tail) with a partial batch. While another stream is
            // active, short batches can use the other free streams.
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

            // Dispatch short batches too: all relaxations run on the GPU.
            if (has_dispatchable_slot() && !queue_.empty()) {
                fill_free_slots(true);
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
        // A paired cooperative grid owns the device until completion.  Do not
        // enqueue an ordinary kernel from another stream while its blocks are
        // using grid-wide barriers.
        for (const auto& active : slots_)
            if (active.in_use && active.paired_dicg) return;
        for (auto& slot : slots_) {
            if (slot.in_use) continue;
            // Normally launch only full batches. The caller may allow one
            // partial batch for a free stream when the frontier is small.
            auto batch = queue_.pop_full_batch(
                cfg_.gpu_batch_size, incumbent_.get());
            if (batch.empty() && allow_partial)
                batch = queue_.pop_batch(
                    cfg_.gpu_batch_size, incumbent_.get());
            if (batch.empty()) break;
            active_bounds_.insert(batch);
            active_tasks_.fetch_add((int)batch.size(), std::memory_order_relaxed);
            submit_slot(slot, std::move(batch));
            if (slot.paired_dicg) break;
            // At most one under-filled batch should be launched. Once it has
            // consumed the frontier there cannot be useful work for another
            // free slot in this pass.
            if (allow_partial) break;
        }
    }

    bool has_dispatchable_slot() const {
        bool free = false;
        for (const auto& slot : slots_) {
            if (slot.in_use && slot.paired_dicg) return false;
            free = free || !slot.in_use;
        }
        return free;
    }

    // Refill an empty frontier in best-bound order using final GPU bounds.
    // Keep parents active until post-processing retires them; their original
    // tracked bounds remain conservative while children are in flight.
    void branch_starved_batches(std::vector<DRSCompletedBatch>& batches) {
        struct Candidate { size_t batch; int node; double bound; };
        std::vector<Candidate> candidates;
        for (size_t b = 0; b < batches.size(); ++b)
            for (int i = 0; i < batches[b].n_nodes; ++i)
                candidates.push_back({b, i, batches[b].lower_bound[i]});
        std::stable_sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.bound < b.bound; });
        for (const auto& candidate : candidates) {
            if (finish_if_time_limit_reached() || !has_dispatchable_slot()) return;
            auto& batch = batches[candidate.batch];
            const int i = candidate.node;
            Node nd = batch.nodes[i];
            nd.dual_bound = candidate.bound;
            if (pruned_early_parents_.count(nd.parent_id) || should_prune(nd)) continue;
            const auto begin = batch.x.begin() + (size_t)i * prob_.n_vars;
            std::vector<double> x(begin, begin + prob_.n_vars);
            if (!std::all_of(x.begin(), x.end(),
                    [](double v) { return std::isfinite(v); })
                    || batch.node_state[i] >= DRSIntegralUnconverged) continue;
            int nid = next_id_.fetch_add(8, std::memory_order_relaxed);
            auto children = branch_sel_(nd, nid, x, prob_.int_vars);
            batch.branched_early[i] = 1;
            for (auto& child : children) {
                child.relaxation_x.clear();
                child.relaxation_iterations = 0;
                if (!should_prune(child)) queue_.push(child);
            }
            if (queue_.size() >= cfg_.gpu_batch_size) {
                // This pass submits only one batch, then resumes host work.
                fill_free_slots(true);
                return;
            }
        }
    }

    void submit_slot(DRSStreamSlot& slot, std::vector<Node> batch) {
        const int n = prob_.n_vars;
        const int N = (int)batch.size();
        slot.paired_dicg = choose_paired_dicg(N, n);
        if (slot.paired_dicg)
            paired_gpu_batches_.fetch_add(1, std::memory_order_relaxed);
        slot.batch_nodes = std::move(batch);
        slot.n_nodes = N;
        slot.in_use = true;
        ++slot.generation;

        for (int i = 0; i < N; i++) {
            const Node& nd = slot.batch_nodes[i];
            for (int j = 0; j < n; j++) {
                slot.h_x [i*n+j] = nd.relaxation_x.empty()
                    ? nd.lb[j] : nd.relaxation_x[j];
                slot.h_lb[i*n+j] = nd.lb[j];
                slot.h_ub[i*n+j] = nd.ub[j];
            }
            slot.h_gap[i] = 0.0;
            slot.h_eps[i] = node_tolerance(nd);
            slot.h_iterations[i] = nd.relaxation_iterations;
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
        // CUDA current device is thread-local; occupancy and streams were
        // configured on the constructing thread's selected device.
        if (multiblock_device_ >= 0) DRS_CUDA_CHECK(cudaSetDevice(multiblock_device_));
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

        if (cfg_.execution_mode == FWExecutionMode::MultiBlock) {
            DRS_CUDA_CHECK(cudaMemsetAsync(slot.d_node_controls, 0,
                drs_multiblock_control_bytes(N, cfg_.blocks_per_node, DRSLMOWorkspace<LMO>::bytes), slot.stream));
            size_t objective_stride = objective_workspace_stride();
            int max_iters = cfg_.gpu_max_iters;
            int launch_N = N;
            void* args[] = {
                &obj_, &alg_, &lmo_, &slot.d_x, &slot.d_lb, &slot.d_ub,
                &slot.d_gap, &slot.d_value, &slot.d_iterations, &slot.d_cnv,
                &slot.d_best_vertex, &slot.d_best_vertex_val,
                &slot.d_workspace, &slot.d_objective_workspace,
                &objective_stride, &launch_N, &max_iters, &slot.d_eps,
                &slot.d_integer_flags, &cfg_.integer_tol, &heuristic_,
                &cfg_.blocks_per_node, &slot.d_node_controls
            };
            DRS_CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)drs_fw_multiblock_kernel<DeviceObjective, Algorithm, LMO, Heuristic>,
                dim3(N * cfg_.blocks_per_node), dim3(block_threads), args,
                smem, slot.stream));
        } else if constexpr (DRSUsesDICG<Algorithm>::value) {
            if (slot.paired_dicg) {
                DRS_CUDA_CHECK(cudaMemsetAsync(
                    slot.d_all_done, 0, sizeof(int), slot.stream));
                size_t objective_stride = objective_workspace_stride();
                int max_iters = cfg_.gpu_max_iters;
                int launch_N = N;
                void* args[] = {
                    &obj_, &alg_, &lmo_, &slot.d_x, &slot.d_lb, &slot.d_ub,
                    &slot.d_gap, &slot.d_value, &slot.d_iterations, &slot.d_cnv,
                    &slot.d_best_vertex, &slot.d_best_vertex_val,
                    &slot.d_workspace, &slot.d_objective_workspace,
                    &objective_stride, &launch_N, &max_iters, &slot.d_eps,
                    &slot.d_all_done, &slot.d_integer_flags, &cfg_.integer_tol,
                    &heuristic_
                };
                DRS_CUDA_CHECK(cudaLaunchCooperativeKernel(
                    (void*)drs_dicg_paired_batch_kernel<
                        DeviceObjective, Algorithm, LMO, Heuristic>,
                    dim3(2 * N), dim3(block_threads), args, smem,
                    slot.stream));
            } else {
                drs_fw_relax_batch_kernel<DeviceObjective, Algorithm, LMO, Heuristic>
                    <<<N, block_threads, smem, slot.stream>>>(
                        obj_, alg_, lmo_, slot.d_x, slot.d_lb, slot.d_ub,
                        slot.d_gap, slot.d_value, slot.d_iterations, slot.d_cnv,
                        slot.d_best_vertex, slot.d_best_vertex_val,
                        slot.d_workspace, slot.d_objective_workspace,
                        objective_workspace_stride(), N, cfg_.gpu_max_iters,
                        slot.d_eps, slot.d_integer_flags, cfg_.integer_tol, heuristic_);
            }
        } else {
            drs_fw_relax_batch_kernel<DeviceObjective, Algorithm, LMO, Heuristic>
                <<<N, block_threads, smem, slot.stream>>>(
                    obj_, alg_, lmo_, slot.d_x, slot.d_lb, slot.d_ub,
                    slot.d_gap, slot.d_value, slot.d_iterations, slot.d_cnv,
                    slot.d_best_vertex, slot.d_best_vertex_val,
                    slot.d_workspace, slot.d_objective_workspace,
                    objective_workspace_stride(), N, cfg_.gpu_max_iters,
                    slot.d_eps, slot.d_integer_flags, cfg_.integer_tol, heuristic_);
        }
        DRS_CUDA_CHECK(cudaGetLastError());

        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.h_x,   slot.d_x,   xn * sizeof(double),  cudaMemcpyDeviceToHost, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.h_gap, slot.d_gap, N  * sizeof(double),  cudaMemcpyDeviceToHost, slot.stream));
        DRS_CUDA_CHECK(cudaMemcpyAsync(slot.h_value, slot.d_value, N * sizeof(double), cudaMemcpyDeviceToHost, slot.stream));
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
        batch.branched_early.assign(N, 0);
        batch.node_state.assign(slot.h_cnv, slot.h_cnv + N);
        batch.gap.assign(slot.h_gap, slot.h_gap + N);
        batch.value.assign(slot.h_value, slot.h_value + N);
        batch.lower_bound.reserve(N);
        for (int i = 0; i < N; ++i)
            batch.lower_bound.push_back(gpuboscia::certified_fw_lower_bound(
                slot.batch_nodes[i].dual_bound, batch.value[i], batch.gap[i]));
        for (int i = 0; i < N; ++i)
            if (batch.node_state[i] == DRSInfeasible)
                batch.lower_bound[i] = std::numeric_limits<double>::infinity();
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
            if (batch.node_state[i] == DRSInfeasible || pruned_early_parents_.count(nd.parent_id)) {
                const int iter = nodes_done_.fetch_add(1, std::memory_order_relaxed) + 1;
                active_tasks_.fetch_sub(1, std::memory_order_acq_rel);
                active_bounds_.erase_one(active_bound);
                emit_progress(iter, false);
                finish_if_search_complete();
                continue;
            }
            std::vector<double> xi(batch.x.begin() + (size_t)i*n,
                                   batch.x.begin() + (size_t)(i+1)*n);

            // Both values refer to the final GPU iterate. No host oracle replay.
            const double gap = batch.gap[i];

            last_gpu_iterations_.store(batch.iterations[i], std::memory_order_relaxed);
            last_gpu_gap_.store(gap, std::memory_order_relaxed);
            last_gpu_converged_.store(
                std::isfinite(gap) && gap >= 0.0
                    && gap < node_tolerance(nd),
                std::memory_order_relaxed);

            nd.dual_bound = batch.lower_bound[i];

            bool new_incumbent = false;
            if (batch.best_vertex_val[i] < 1e299) {
                std::vector<double> best_vertex(
                    batch.best_vertex.begin() + (size_t)i*n,
                    batch.best_vertex.begin() + (size_t)(i+1)*n);
                if (std::isfinite(batch.best_vertex_val[i]))
                    new_incumbent =
                        incumbent_.try_update(batch.best_vertex_val[i], best_vertex);
            }

            const bool certificate_valid =
                std::isfinite(gap) && gap >= 0.0
                && gap < node_tolerance(nd);

            if (should_prune(nd)) {
                if (batch.branched_early[i]) {
                    pruned_early_parents_.insert(nd.id);
                    queue_.discard_children(nd.id);
                }
                const int iter = nodes_done_.fetch_add(1, std::memory_order_relaxed) + 1;
                active_tasks_.fetch_sub(1, std::memory_order_acq_rel);
                active_bounds_.erase_one(active_bound);
                emit_progress(iter, new_incumbent);
                finish_if_search_complete();
                continue;
            }

            if (batch.node_state[i] >= DRSIntegralUnconverged
                    && !certificate_valid) {
                // Keep this node live, with its certified bound and GPU state.
                nd.relaxation_x = std::move(xi);
                nd.relaxation_iterations = batch.iterations[i];
                queue_.push(nd);
                active_tasks_.fetch_sub(1, std::memory_order_acq_rel);
                active_bounds_.erase_one(active_bound);
                emit_progress(nodes_done_.load(), new_incumbent);
                continue;
            }

            if (done_.load(std::memory_order_relaxed)) {
                // Stop expanding this completed batch once another path has
                // proven the search complete.
            } else if (batch.node_state[i] >= DRSIntegralUnconverged) {
                // The GPU heuristic already tested the strict
                // integer candidate. Do not store the near-integer raw point.
            } else if (!batch.branched_early[i]) {
                int nid = next_id_.fetch_add(8, std::memory_order_relaxed);
                for (auto& child : branch_sel_(nd, nid, xi, prob_.int_vars)) {
                    child.relaxation_x.clear();
                    child.relaxation_iterations = 0;
                    if (!should_prune(child)) queue_.push(child);
                }
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
        if (gpuboscia::gap_within_tolerance(
                bound, incumbent, cfg_.abs_gap_tol, cfg_.rel_gap_tol)) {
            final_dual_bound_ = bound;
            reached_tolerance_ = true;
            signal_done();
            return true;
        }
        return false;
    }

    double node_tolerance(const Node& nd) const {
        return gpuboscia::depth_dependent_tolerance(
            cfg_.gpu_eps, cfg_.gpu_eps_decay, cfg_.gpu_eps_min, nd.depth);
    }

    bool should_prune(const Node& nd) const {
        return nd.dual_bound >= incumbent_.get();
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
