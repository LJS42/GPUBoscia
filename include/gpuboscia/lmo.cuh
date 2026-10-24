#pragma once
// Linear Minimization Oracle (LMO) interface for Frank-Wolfe on GPU.
//
// LMO is a public extension point.  Users can pass their own LMO type as the
// third template argument to DeviceResidentHybridSolver.
// The type is copied into kernels, so keep it trivially copyable or at least
// CUDA-kernel-copyable: store small scalar parameters, fixed-size arrays, or
// device pointers.  Do not store std::vector or host-only owning containers in
// an LMO object used by CUDA kernels.
//
// An LMO struct must provide one device method:
//
//   __device__ void compute_v(int tid, int n,
//                              const double* grad,   // [n] gradient (shared mem)
//                              const double* lb,     // [n] lower bounds (device)
//                              const double* ub,     // [n] upper bounds (device)
//                              double*       v) const; // [n] output vertex (shared mem)
//
// Called with blockDim.x threads; tid is threadIdx.x. n may exceed blockDim.x,
// so coordinate-parallel implementations must iterate with
// `for (int j = tid; j < n; j += blockDim.x)`. The implementation may instead
// use only tid == 0, but after the method returns every v[j] for j in [0, n)
// must be initialized. A __syncthreads() after the call is the caller's
// responsibility.
//
// Solvers also call compute_v once with a deterministic, non-zero artificial
// direction to obtain a feasible initial FW iterate.  Therefore compute_v must
// return a feasible point for every finite direction, not only for objective
// gradients encountered during the main FW loop.
// An optional explicit root iterate bypasses this initialization. To validate
// custom continuous constraints on the host, provide:
//   bool relaxation_feasible_cpu(const std::vector<double>& x,
//       const std::vector<double>& lb, const std::vector<double>& ub,
//       double tolerance) const;
// This must check continuous membership, not integrality of an incumbent.
//
// A DICG-compatible LMO additionally provides compute_inface_v(),
// dicg_maximum_step_local(), and dicg_sum_maximum_step().  BoxLMO and
// UnitSimplexLMO implement this extended interface.
//
// Node feasibility is checked in the relaxation block before objective work.
// Every LMO provides a cooperative check returning a block-uniform bool:
//   __device__ bool check_feasible_device(int tid, int n,
//       const double* lb, const double* ub, const std::uint8_t* integer_flags,
//       double* vertex_scratch, double* direction_scratch) const;
// Common GPU checks validate bounds and integer intervals. The LMO checks
// its additional constraints; both scratch vectors have n elements.
//
// Minimal custom LMO shape:
//
//   struct MyLMO {
//       double param;
//
//       __device__ bool check_feasible_device(int, int,
//           const double*, const double*, const std::uint8_t*, double*, double*) const {
//           return true; // Only for a box; check custom constraints here.
//       }
//
//       __device__ void compute_v(int tid, int n,
//                                 const double* grad,
//                                 const double* lb,
//                                 const double* ub,
//                                 double* v) const {
//           for (int j = tid; j < n; j += blockDim.x) v[j] = ...;
//       }
//   };

// ---------------------------------------------------------------
//  BoxLMO — argmin_{v ∈ [lb, ub]} <grad, v>
//  Solution: v[j] = lb[j] if grad[j] > 0, else ub[j]
// ---------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#ifdef __CUDACC__
#include <math_constants.h>
#endif
#include "bnb.h"
#include "lmo_dicg.h"
#include "node.h"

#ifdef __CUDACC__
// Same algorithm can use block-local or node-wide scratch and barriers.
struct LMOBlockContext {
    int global_tid, total_threads;
    __device__ LMOBlockContext() : global_tid(threadIdx.x), total_threads(blockDim.x) {}
    __device__ void node_sync() const { __syncthreads(); }
};
// Cooperative sum with a block-uniform result, including partial warps.
__device__ inline double feasibility_block_sum(double value) {
    __shared__ double partial[32];
    __shared__ double total;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const unsigned mask = __activemask();
    const int lanes = __popc(mask);
    for (int offset = 16; offset > 0; offset >>= 1) {
        const double other = __shfl_down_sync(mask, value, offset);
        if (lane + offset < lanes) value += other;
    }
    if (lane == 0) partial[tid >> 5] = value;
    __syncthreads();
    if (tid < 32) {
        const int warps = (blockDim.x + 31) >> 5;
        value = tid < warps ? partial[tid] : 0.0;
        const unsigned first_mask = __activemask();
        const int first_lanes = __popc(first_mask);
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double other = __shfl_down_sync(first_mask, value, offset);
            if (lane + offset < first_lanes) value += other;
        }
        if (tid == 0) total = value;
    }
    __syncthreads();
    return total;
}
#endif

struct LmoMinPair {
    double value;
    int    index;
};

__device__ inline LmoMinPair lmo_better_min_pair(LmoMinPair a, LmoMinPair b)
{
    if (b.index < 0)
        return a;
    if (a.index < 0)
        return b;
    if (b.value < a.value)
        return b;
    if (b.value == a.value && b.index < a.index)
        return b;
    return a;
}

__device__ inline LmoMinPair lmo_warp_min_pair(LmoMinPair p)
{
    for (int off = 16; off > 0; off >>= 1) {
        LmoMinPair other{
            __shfl_down_sync(0xffffffff, p.value, off),
            __shfl_down_sync(0xffffffff, p.index, off)
        };
        p = lmo_better_min_pair(p, other);
    }
    return p;
}

__device__ inline LmoMinPair lmo_block_min_pair(LmoMinPair p, int tid)
{
    __shared__ double warp_values[32];
    __shared__ int    warp_indices[32];

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;

    p = lmo_warp_min_pair(p);
    if (lane == 0) {
        warp_values[warp] = p.value;
        warp_indices[warp] = p.index;
    }
    __syncthreads();

    LmoMinPair block_p{0.0, -1};
    if (warp == 0) {
        block_p = (lane < nwarps)
            ? LmoMinPair{warp_values[lane], warp_indices[lane]}
            : LmoMinPair{0.0, -1};
        block_p = lmo_warp_min_pair(block_p);
    }
    if (tid == 0) {
        warp_values[0] = block_p.value;
        warp_indices[0] = block_p.index;
    }
    __syncthreads();
    return LmoMinPair{warp_values[0], warp_indices[0]};
}

__device__ inline double lmo_warp_sum(double v)
{
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ inline double lmo_block_sum(double v, int tid)
{
    __shared__ double warp_sums[32];

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;

    v = lmo_warp_sum(v);
    if (lane == 0)
        warp_sums[warp] = v;
    __syncthreads();

    double total = 0.0;
    if (warp == 0) {
        total = (lane < nwarps) ? warp_sums[lane] : 0.0;
        total = lmo_warp_sum(total);
    }
    if (tid == 0)
        warp_sums[0] = total;
    __syncthreads();
    return warp_sums[0];
}

struct BoxLMO {
    static constexpr bool supports_dicg = true;
    bool relaxation_feasible_cpu(const std::vector<double>&,
        const std::vector<double>&, const std::vector<double>&, double) const {
        return true; // The public API independently validates all box bounds.
    }


    __device__ bool check_feasible_device(int, int, const double*,
        const double*, const std::uint8_t*, double*, double*) const {
        return true; // Common GPU bounds/integer-interval check handles boxes.
    }

    void compute_v_cpu(
        const std::vector<double>& grad,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<double>&       v) const
    {
        const int n = (int)grad.size();
        for (int j = 0; j < n; j++)
            v[j] = (grad[j] > 0.0) ? lb[j] : ub[j];
    }

    // DICG in-face oracle.  Coordinates already on a bound are fixed because
    // they define the minimal face containing x; all other coordinates are
    // minimized exactly as in the forward box LMO.
    void compute_inface_v_cpu(
        const std::vector<double>& grad,
        const std::vector<double>& x,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<double>&       a,
        double                     atol = 1e-6,
        double                     rtol = 1e-4) const
    {
        box_inface_lmo_cpu(grad, x, lb, ub, a, atol, rtol);
    }

    // Maximum gamma for x - gamma * direction to remain in [lb, ub].
    // In DICG, direction is a - v, hence the actual step is gamma * (v - a).
    double dicg_maximum_step_cpu(
        const std::vector<double>& direction,
        const std::vector<double>& x,
        const std::vector<double>& lb,
        const std::vector<double>& ub) const
    {
        return box_dicg_maximum_step_cpu(direction, x, lb, ub);
    }

    // All K blocks participate; the caller publishes output with a node barrier.
    template<class Context>
    __device__ void compute_v_multiblock(Context& ctx, int n,
        const double* grad, const double* lb, const double* ub, double* v) const {
        for (int j = ctx.global_tid; j < n; j += ctx.total_threads)
            v[j] = (grad[j] > 0.0) ? lb[j] : ub[j];
    }

    __device__ void compute_v(int tid, int n,
                               const double* grad,
                               const double* lb, const double* ub,
                               double* v) const {
        for (int j = tid; j < n; j += blockDim.x)
            v[j] = (grad[j] > 0.0) ? lb[j] : ub[j];
    }

    __device__ void compute_inface_v(
        int tid, int n,
        const double* grad, const double* x,
        const double* lb, const double* ub,
        double* a,
        double atol = 1e-6, double rtol = 1e-4) const
    {
        for (int j = tid; j < n; j += blockDim.x) {
            const double upper_tol = atol + rtol * fabs(ub[j]);
            const double lower_tol = atol + rtol * fabs(lb[j]);
            if (fabs(x[j] - ub[j]) <= upper_tol)
                a[j] = ub[j];
            else if (fabs(x[j] - lb[j]) <= lower_tol)
                a[j] = lb[j];
            else
                a[j] = (grad[j] > 0.0) ? lb[j] : ub[j];
        }
    }

    // Per-coordinate limit for x - gamma * direction.  The DICG kernel can
    // obtain the global maximum step with a block minimum reduction.
    __device__ double dicg_maximum_step_local(
        int tid, int n,
        const double* direction, const double* x,
        const double* lb, const double* ub) const
    {
        double result = 1.0;
        for (int j = tid; j < n; j += blockDim.x) {
            const double d = direction[j];
            if (d < 0.0)
                result = fmin(result, (ub[j] - x[j]) / -d);
            else if (d > 0.0)
                result = fmin(result, (x[j] - lb[j]) / d);
        }
        return result;
    }

    __device__ double dicg_sum_maximum_step(
        int /*tid*/, int /*n*/,
        const double* /*direction*/, const double* /*x*/) const
    {
        return 1.0;
    }
};

// ---------------------------------------------------------------
//  UnitSimplexLMO — Julia-equivalent bounded simplex oracle.
//
//  The integer-variable lower bounds are mandatory.  Remaining simplex mass
//  is assigned greedily to variables with non-positive gradient, in ascending
//  gradient order.  Integer variables are capped by ub[j] - lb[j]; continuous
//  variables receive all remaining mass.
//
//  Feasibility follows the same model: the mandatory integer lower-bound mass
//  must not exceed right_side, and integer-variable bound intervals must be
//  non-empty.
// ---------------------------------------------------------------
template<int MaxN, bool RequireEquality = false>
struct UnitSimplexLMO {
    static constexpr bool supports_dicg = true;

    static constexpr int next_power_of_two(int value)
    {
        int result = 1;
        while (result < value)
            result <<= 1;
        return result;
    }

    static constexpr int sort_size = next_power_of_two(MaxN);

    double right_side = 1.0;
    unsigned char is_int[MaxN]{};

    UnitSimplexLMO() = default;

    UnitSimplexLMO(
        double                  rhs,
        int                     n,
        const std::vector<int>& int_vars)
        : right_side(rhs)
    {
        set_integer_vars(n, int_vars);
    }

    void set_integer_vars(int n, const std::vector<int>& int_vars)
    {
        std::fill(is_int, is_int + MaxN, (unsigned char)0);
        const int capped_n = std::min(n, MaxN);
        for (int idx : int_vars) {
            if (idx >= 0 && idx < capped_n)
                is_int[idx] = 1;
        }
    }

    __device__ bool check_feasible_device(int tid, int n, const double* lb,
        const double* ub, const std::uint8_t* flags, double*, double*) const {
        if (n > MaxN) return false;
        bool valid = true, continuous = false;
        double mandatory = 0.0, capacity = 0.0;
        for (int j = tid; j < n; j += blockDim.x) {
            const bool integer = flags ? flags[j] != 0 : is_int[j] != 0;
            const double lo = integer ? ceil(fmax(0.0, lb[j])) : fmax(0.0, lb[j]);
            const double hi = integer ? floor(ub[j]) : ub[j];
            valid &= lo <= hi;
            mandatory += lo;
            capacity += hi;
            continuous |= !integer;
        }
        const bool intervals_valid = __syncthreads_and(valid);
        const double mandatory_sum = feasibility_block_sum(mandatory);
        bool feasible = intervals_valid && mandatory_sum <= right_side + 1e-9;
        if constexpr (RequireEquality) {
            const double capacity_sum = feasibility_block_sum(capacity);
            const bool has_continuous = __syncthreads_or(continuous);
            feasible &= capacity_sum >= right_side - 1e-9
                && (has_continuous || fabs(right_side - round(right_side)) <= 1e-9);
        }
        return feasible;
    }

    void compute_v_cpu(
        const std::vector<double>& grad,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<double>&       v) const
    {
        const int n = (int)grad.size();
        const int capped_n = std::min(n, MaxN);
        std::vector<unsigned char> used(capped_n, 0);

        double mass = 0.0;
        for (int j = 0; j < capped_n; j++) {
            if (is_int[j]) {
                v[j] = lb[j];
                mass += v[j];
            } else {
                v[j] = 0.0;
            }
        }
        for (int j = capped_n; j < n; j++)
            v[j] = 0.0;

        double remaining = right_side - mass;
        while (remaining > 0.0) {
            int best = -1;
            double best_grad = 0.0;
            for (int j = 0; j < capped_n; j++) {
                if (used[j] || (!RequireEquality && grad[j] > 0.0))
                    continue;
                if (best < 0 || grad[j] < best_grad
                        || (grad[j] == best_grad && j < best)) {
                    best = j;
                    best_grad = grad[j];
                }
            }
            if (best < 0)
                break;

            used[best] = 1;
            double add = remaining;
            if (is_int[best]) {
                const double available =
                    std::max(ub[best] - lb[best], 0.0);
                add = RequireEquality
                    ? std::min(available, std::floor(remaining + 1e-9))
                    : std::min(available, remaining);
            }

            v[best] += add;
            remaining -= add;
        }
    }

    void compute_inface_v_cpu(
        const std::vector<double>& grad,
        const std::vector<double>& x,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<double>&       a,
        double                     atol = 1e-6,
        double                     rtol = 1e-4) const
    {
        const int n = std::min((int)grad.size(), MaxN);
        std::vector<unsigned char> integer_flags(
            is_int, is_int + n);
        unit_simplex_inface_lmo_cpu(
            right_side, integer_flags, grad, x, lb, ub, a, atol, rtol,
            RequireEquality);
    }

    double dicg_maximum_step_cpu(
        const std::vector<double>& direction,
        const std::vector<double>& x,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        double                     tol = 1e-6) const
    {
        const int n = std::min((int)direction.size(), MaxN);
        std::vector<unsigned char> integer_flags(
            is_int, is_int + n);
        return unit_simplex_dicg_maximum_step_cpu(
            right_side, integer_flags, direction, x, lb, ub, tol,
            RequireEquality);
    }

    bool relaxation_feasible_cpu(const std::vector<double>& x,
        const std::vector<double>&, const std::vector<double>&, double tol) const {
        return dicg_linear_feasible_cpu(x, tol);
    }

    bool dicg_linear_feasible_cpu(
        const std::vector<double>& v, double tol = 1e-3) const
    {
        if constexpr (RequireEquality)
            return equality_simplex_linear_feasible_cpu(right_side, v, tol);
        return unit_simplex_linear_feasible_cpu(right_side, v, tol);
    }

    bool dicg_inface_feasible_cpu(
        const std::vector<double>& a,
        const std::vector<double>& x,
        double                     atol = 1e-6,
        double                     rtol = 1e-4) const
    {
        if constexpr (RequireEquality)
            return equality_simplex_linear_feasible_cpu(
                right_side, a, atol + rtol * std::abs(right_side));
        return unit_simplex_inface_feasible_cpu(
            right_side, a, x, atol, rtol);
    }

    // Parallel fractional-knapsack allocation shared by the forward and
    // in-face simplex oracles.  Eligible coordinates are sorted by
    // (gradient, index), their capacities are exclusive-scanned, and every
    // output coordinate is then filled independently from its prefix mass.
    // Padding to sort_size lets the block use a regular bitonic network even
    // when MaxN is not a power of two.  Threads may process multiple network
    // comparators, so sort_size need not be <= blockDim.x.
    __device__ void parallel_greedy_allocate(
        int, int capped_n, const double* grad, const double* ub,
        double* out, const unsigned char* fixed, double remaining,
        bool force_full_mass, bool floor_integer_mass) const {
        __shared__ double sort_keys[sort_size], prefix[sort_size];
        __shared__ int sort_indices[sort_size];
        LMOBlockContext ctx;
        parallel_greedy_allocate_cooperative(ctx, sort_keys, sort_indices, prefix,
            capped_n, grad, ub, out, fixed, remaining, force_full_mass, floor_integer_mass);
    }

    template<class Context>
    __device__ void parallel_greedy_allocate_cooperative(Context& ctx,
        double* sort_keys, int* sort_indices, double* prefix,
        int capped_n, const double* grad, const double* ub, double* out,
        const unsigned char* fixed, double remaining,
        bool force_full_mass, bool floor_integer_mass) const {
        const int tid = ctx.global_tid;
        for (int pos = tid; pos < sort_size; pos += ctx.total_threads) {
            const bool eligible = pos < capped_n
                && (!fixed || !fixed[pos])
                && (force_full_mass || grad[pos] <= 0.0);
            sort_keys[pos] = eligible ? grad[pos] : CUDART_INF;
            sort_indices[pos] = eligible ? pos : -1;
        }
        ctx.node_sync();

        // Stable bitonic sort by (gradient, original coordinate).  Padding
        // entries have index -1 and an infinite key, and stay at the end.
        for (int width = 2; width <= sort_size; width <<= 1) {
            for (int stride = width >> 1; stride > 0; stride >>= 1) {
                for (int left = tid; left < sort_size;
                     left += ctx.total_threads) {
                    const int right = left ^ stride;
                    if (right <= left)
                        continue;

                    const double lk = sort_keys[left];
                    const double rk = sort_keys[right];
                    const int li = sort_indices[left];
                    const int ri = sort_indices[right];
                    const bool left_after_right = rk < lk
                        || (rk == lk && ri >= 0 && (li < 0 || ri < li));
                    const bool ascending = (left & width) == 0;
                    if (left_after_right == ascending) {
                        sort_keys[left] = rk;
                        sort_keys[right] = lk;
                        sort_indices[left] = ri;
                        sort_indices[right] = li;
                    }
                }
                ctx.node_sync();
            }
        }

        // Capacity in sorted order.  A continuous coordinate can consume all
        // remaining mass, while an integer coordinate is bounded by its
        // residual branch interval.
        for (int pos = tid; pos < sort_size; pos += ctx.total_threads) {
            const int j = sort_indices[pos];
            prefix[pos] = j < 0 ? 0.0
                : (is_int[j] ? fmax(ub[j] - out[j], 0.0) : remaining);
        }
        ctx.node_sync();

        // In-place Blelloch exclusive scan of the capacities.
        for (int offset = 1; offset < sort_size; offset <<= 1) {
            for (int op = tid; op < sort_size / (offset << 1);
                 op += ctx.total_threads) {
                const int right = (op + 1) * (offset << 1) - 1;
                prefix[right] += prefix[right - offset];
            }
            ctx.node_sync();
        }
        if (tid == 0)
            prefix[sort_size - 1] = 0.0;
        ctx.node_sync();
        for (int offset = sort_size >> 1; offset > 0; offset >>= 1) {
            for (int op = tid; op < sort_size / (offset << 1);
                 op += ctx.total_threads) {
                const int right = (op + 1) * (offset << 1) - 1;
                const double left_sum = prefix[right - offset];
                prefix[right - offset] = prefix[right];
                prefix[right] += left_sum;
            }
            ctx.node_sync();
        }

        for (int pos = tid; pos < sort_size; pos += ctx.total_threads) {
            const int j = sort_indices[pos];
            if (j < 0)
                continue;
            double add = fmax(remaining - prefix[pos], 0.0);
            const double capacity = is_int[j]
                ? fmax(ub[j] - out[j], 0.0) : remaining;
            add = fmin(add, capacity);
            if (floor_integer_mass && is_int[j])
                add = floor(add + (RequireEquality ? 1e-9 : 0.0));
            out[j] += add;
        }
        ctx.node_sync();
    }

    // Tiles remain in shared memory through all local sorting/scan stages.
    // Only completed sorted runs and tile summaries cross block boundaries.
    static constexpr int multiblock_tile_size = 128;
    static constexpr int multiblock_tiles =
        (sort_size + multiblock_tile_size - 1) / multiblock_tile_size;
    struct MultiBlockWorkspace {
        double keys[sort_size], merged_keys[sort_size], prefix[sort_size];
        int indices[sort_size], merged_indices[sort_size];
        double mass[multiblock_tiles], tile_sum[multiblock_tiles];
    };
    struct TileWorkspace {
        double keys[multiblock_tile_size], scan[multiblock_tile_size];
        int indices[multiblock_tile_size];
    };
    static constexpr size_t multiblock_workspace_bytes = sizeof(MultiBlockWorkspace);
    __host__ __device__ static int multiblock_effective_blocks(int blocks, int) {
        return sort_size <= multiblock_tile_size ? 1
            : (blocks < multiblock_tiles ? blocks : multiblock_tiles);
    }


    __device__ static bool key_before(double a, int ai, double b, int bi) {
        return a < b || (a == b && ai >= 0 && (bi < 0 || ai < bi));
    }

    template<class Context>
    __device__ void compute_v_multiblock(Context& ctx, int n,
        const double* grad, const double* lb, const double* ub, double* out) const {
        if (ctx.block_count == 1 || sort_size <= multiblock_tile_size) {
            if (ctx.block_rank == 0) compute_v(ctx.thread_rank, n, grad, lb, ub, out);
            return;
        }
        __shared__ TileWorkspace tile;
        compute_v_hierarchical(ctx, tile, n, grad, lb, ub, out);
    }

    template<class Context>
    __device__ void compute_v_hierarchical(Context& ctx, TileWorkspace& tile, int n,
        const double* grad, const double* lb, const double* ub, double* out) const {
        auto& ws = *reinterpret_cast<MultiBlockWorkspace*>(ctx.lmo_workspace);
        const int tid = ctx.thread_rank;
        const int capped_n = n < MaxN ? n : MaxN;
        constexpr int tile_size = multiblock_tile_size;
        for (int j = ctx.global_tid; j < n; j += ctx.total_threads)
            out[j] = j < capped_n && is_int[j] ? lb[j] : 0.0;
        for (int t = ctx.block_rank; t < multiblock_tiles; t += ctx.block_count) {
            for (int j = tid; j < tile_size; j += ctx.threads_per_block) {
                const int coord = t*tile_size+j;
                const bool eligible = coord < capped_n && (RequireEquality || grad[coord] <= 0.0);
                tile.keys[j] = eligible ? grad[coord] : CUDART_INF;
                tile.indices[j] = eligible ? coord : -1;
                tile.scan[j] = coord < capped_n && is_int[coord] ? lb[coord] : 0.0;
            }
            ctx.block_sync();
            for (int stride = tile_size/2; stride; stride >>= 1) {
                for (int j = tid; j < stride; j += ctx.threads_per_block)
                    tile.scan[j] += tile.scan[j+stride];
                ctx.block_sync();
            }
            if (tid == 0) ws.mass[t] = tile.scan[0];
            for (int width = 2; width <= tile_size; width <<= 1) {
                for (int stride = width/2; stride; stride >>= 1) {
                    for (int left = tid; left < tile_size; left += ctx.threads_per_block) {
                        const int right = left ^ stride;
                        if (right <= left) continue;
                        if (key_before(tile.keys[right], tile.indices[right],
                                       tile.keys[left], tile.indices[left]) == ((left & width) == 0)) {
                            const double key = tile.keys[left];
                            const int index = tile.indices[left];
                            tile.keys[left] = tile.keys[right]; tile.indices[left] = tile.indices[right];
                            tile.keys[right] = key; tile.indices[right] = index;
                        }
                    }
                    ctx.block_sync();
                }
            }
            for (int j = tid; j < tile_size; j += ctx.threads_per_block) {
                ws.keys[t*tile_size+j] = tile.keys[j];
                ws.indices[t*tile_size+j] = tile.indices[j];
            }
            ctx.block_sync(); // The next tile reuses the shared arrays.
        }
        ctx.node_sync();
        double remaining = right_side;
        for (int t = 0; t < multiblock_tiles; ++t) remaining -= ws.mass[t];

        double* keys = ws.keys;
        double* merged_keys = ws.merged_keys;
        int* indices = ws.indices;
        int* merged_indices = ws.merged_indices;
        // Each element binary-searches the other sorted run for its destination.
        // Equal padding keys use original run position to give every writer a
        // unique destination; real gradient ties use original coordinate index.
        for (int width = tile_size; width < sort_size; width <<= 1) {
            for (int pos = ctx.global_tid; pos < sort_size; pos += ctx.total_threads) {
                const int pair_start = (pos / (2*width)) * (2*width);
                const bool right_run = pos >= pair_start + width;
                const int own_start = pair_start + (right_run ? width : 0);
                const int other_start = pair_start + (right_run ? 0 : width);
                const double key = keys[pos];
                const int index = indices[pos];
                int lo = other_start, hi = other_start + width;
                while (lo < hi) {
                    const int mid = (lo+hi)/2;
                    const bool equal = !key_before(keys[mid], indices[mid], key, index)
                        && !key_before(key, index, keys[mid], indices[mid]);
                    if (key_before(keys[mid], indices[mid], key, index) || (equal && right_run)) lo = mid+1;
                    else hi = mid;
                }
                const int dest = pair_start + (pos-own_start) + (lo-other_start);
                merged_keys[dest] = key;
                merged_indices[dest] = index;
            }
            ctx.node_sync();
            auto old_keys = keys; keys = merged_keys; merged_keys = old_keys;
            auto old_indices = indices; indices = merged_indices; merged_indices = old_indices;
        }
        for (int t = ctx.block_rank; t < multiblock_tiles; t += ctx.block_count) {
            for (int j = tid; j < tile_size; j += ctx.threads_per_block) {
                const int coord = indices[t*tile_size+j];
                // Read immutable bounds rather than the output buffer.
                tile.scan[j] = coord < 0 ? 0.0 : (is_int[coord] ? fmax(ub[coord]-lb[coord], 0.0) : remaining);
            }
            ctx.block_sync();
            for (int offset = 1; offset < tile_size; offset <<= 1) {
                for (int op = tid; op < tile_size/(2*offset); op += ctx.threads_per_block) {
                    const int right = (op+1)*2*offset-1;
                    tile.scan[right] += tile.scan[right-offset];
                }
                ctx.block_sync();
            }
            if (tid == 0) {
                ws.tile_sum[t] = tile.scan[tile_size-1];
                tile.scan[tile_size-1] = 0;
            }
            ctx.block_sync();
            for (int offset = tile_size/2; offset; offset >>= 1) {
                for (int op = tid; op < tile_size/(2*offset); op += ctx.threads_per_block) {
                    const int right = (op+1)*2*offset-1;
                    const double left = tile.scan[right-offset];
                    tile.scan[right-offset] = tile.scan[right];
                    tile.scan[right] += left;
                }
                ctx.block_sync();
            }
            for (int j = tid; j < tile_size; j += ctx.threads_per_block)
                ws.prefix[t*tile_size+j] = tile.scan[j];
            ctx.block_sync();
        }
        ctx.node_sync();
        // Tile offsets need only the small summary array; no node-wide scan.
        for (int t = ctx.block_rank; t < multiblock_tiles; t += ctx.block_count) {
            double before = 0;
            for (int prev = 0; prev < t; ++prev) before += ws.tile_sum[prev];
            for (int j = tid; j < tile_size; j += ctx.threads_per_block) {
                const int pos = t*tile_size+j, coord = indices[pos];
                if (coord < 0) continue;
                const double capacity = is_int[coord] ? fmax(ub[coord]-lb[coord], 0.0) : remaining;
                double add = fmin(fmax(remaining-(before+ws.prefix[pos]), 0.0), capacity);
                if (RequireEquality && is_int[coord]) add = floor(add+1e-9);
                out[coord] = (is_int[coord] ? lb[coord] : 0.0) + add;
            }
        }
        // The caller's barrier publishes output and protects scratch reuse.
    }


    __device__ void compute_v(int tid, int n,
                               const double* grad,
                               const double* lb, const double* ub,
                               double* v) const {
        const int capped_n = (n < MaxN) ? n : MaxN;
        __shared__ double remaining_s;

        double local_mass = 0.0;
        for (int j = tid; j < capped_n; j += blockDim.x) {
            if (is_int[j]) {
                v[j] = lb[j];
                local_mass += v[j];
            } else {
                v[j] = 0.0;
            }
        }
        for (int j = capped_n + tid; j < n; j += blockDim.x)
            v[j] = 0.0;
        const double mass = lmo_block_sum(local_mass, tid);
        if (tid == 0)
            remaining_s = right_side - mass;
        __syncthreads();
        parallel_greedy_allocate(
            tid, capped_n, grad, ub, v, nullptr, remaining_s,
            RequireEquality, RequireEquality);
    }

    __device__ void compute_inface_v(
        int tid, int n,
        const double* grad, const double* x,
        const double* lb, const double* ub,
        double* a,
        double atol = 1e-6, double rtol = 1e-4) const
    {
        const int capped_n = (n < MaxN) ? n : MaxN;
        __shared__ unsigned char fixed[MaxN];
        __shared__ double remaining_s;
        __shared__ bool force_full_mass_s;

        double local_mass = 0.0;
        double local_x_mass = 0.0;
        for (int j = tid; j < capped_n; j += blockDim.x) {
            fixed[j] = false;
            a[j] = is_int[j] ? lb[j] : 0.0;

            const double lower = is_int[j] ? lb[j] : 0.0;
            const double lower_tol = atol
                + rtol * fmax(fabs(x[j]), fabs(lower));
            if (fabs(x[j] - lower) <= lower_tol) {
                fixed[j] = true;
                a[j] = lower;
            } else if (is_int[j]) {
                const double upper_tol = atol
                    + rtol * fmax(fabs(x[j]), fabs(ub[j]));
                if (fabs(x[j] - ub[j]) <= upper_tol) {
                    fixed[j] = true;
                    a[j] = ub[j];
                }
            }
            local_mass += a[j];
            local_x_mass += x[j];
        }
        for (int j = capped_n + tid; j < n; j += blockDim.x)
            a[j] = 0.0;

        const double mass = lmo_block_sum(local_mass, tid);
        const double x_mass = lmo_block_sum(local_x_mass, tid);
        if (tid == 0) {
            remaining_s = right_side - mass;
            force_full_mass_s = RequireEquality || fabs(x_mass - right_side)
                <= atol + rtol * fmax(fabs(x_mass), fabs(right_side));
        }
        __syncthreads();
        parallel_greedy_allocate(
            tid, capped_n, grad, ub, a, fixed, remaining_s,
            force_full_mass_s, true);
    }

    __device__ double dicg_maximum_step_local(
        int tid, int n,
        const double* direction, const double* x,
        const double* lb, const double* ub,
        double tol = 1e-6) const
    {
        double result = 1.0;
        const int capped_n = n < MaxN ? n : MaxN;
        for (int j = tid; j < capped_n; j += blockDim.x) {
            const double d = direction[j];
            if (d > tol) {
                const double lower = is_int[j] ? lb[j] : 0.0;
                result = fmin(result, (x[j] - lower) / d);
            } else if (d < -tol) {
                const double upper = is_int[j] ? ub[j] : right_side;
                result = fmin(result, (upper - x[j]) / -d);
            }
        }
        return result;
    }

    // Collective aggregate constraint for x - gamma*direction.  All threads
    // in the block must call this method.
    __device__ double dicg_sum_maximum_step(
        int tid, int n,
        const double* direction, const double* x) const
    {
        double local_x = 0.0;
        double local_direction = 0.0;
        for (int j = tid; j < n; j += blockDim.x) {
            local_x += x[j];
            local_direction += direction[j];
        }
        const double sum_x = lmo_block_sum(local_x, tid);
        const double sum_direction =
            lmo_block_sum(local_direction, tid);
        if (RequireEquality && fabs(sum_direction) > 1e-9)
            return 0.0;
        return sum_direction < 0.0
            ? fmin(1.0, (sum_x - right_side) / sum_direction)
            : 1.0;
    }
};

// Equality-constrained bounded simplex:
// {x >= 0, sum(x) == right_side}, with branch bounds on integer variables.
template<int MaxN>
using EqualitySimplexLMO = UnitSimplexLMO<MaxN, true>;

// Linear minimization over the Order x Order Birkhoff polytope. Variables
// are stored in row-major order. A node lower bound above 1/2 forces an edge;
// an upper bound below 1/2 forbids it. The remaining assignment is solved
// exactly with the Hungarian algorithm.
template<int Order>
struct BirkhoffLMO {
    static_assert(Order > 0, "Birkhoff order must be positive");
    static constexpr bool supports_dicg = true;
    static constexpr int variable_count = Order * Order;
    bool relaxation_feasible_cpu(const std::vector<double>& x,
        const std::vector<double>&, const std::vector<double>&, double tol) const {
        if (x.size() != static_cast<size_t>(variable_count)) return false;
        for (double value : x) if (value < -tol || value > 1 + tol) return false;
        for (int i = 0; i < Order; ++i) {
            double row = 0, column = 0;
            for (int j = 0; j < Order; ++j) {
                row += x[i * Order + j]; column += x[j * Order + i];
            }
            if (std::abs(row - 1) > tol || std::abs(column - 1) > tol) return false;
        }
        return true;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    static constexpr int next_power_of_two(int value) {
        int result = 1;
        while (result < value) result <<= 1;
        return result;
    }
    static constexpr int reduction_size = next_power_of_two(Order);

    __device__ bool check_feasible_device(int tid, int n, const double* lb,
        const double* ub, const std::uint8_t*, double* vertex, double* direction) const {
        if (n != variable_count) return false;
        for (int j = tid; j < n; j += blockDim.x) {
            direction[j] = 0.0;
            vertex[j] = 0.0;
        }
        __syncthreads();
        return solve_device(tid, direction, lb, ub, vertex, nullptr);
    }

    void compute_v_cpu(
        const std::vector<double>& grad,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<double>& v) const
    {
        if ((int)v.size() != variable_count)
            v.resize(variable_count);
        if (!solve_cpu(grad, lb, ub, v, nullptr))
            std::fill(v.begin(), v.end(), 0.0);
    }

    void compute_inface_v_cpu(
        const std::vector<double>& direction,
        const std::vector<double>& x,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<double>& v) const
    {
        if ((int)v.size() != variable_count) v.resize(variable_count);
        if (!solve_cpu(direction, lb, ub, v, &x))
            std::fill(v.begin(), v.end(), 0.0);
    }

    double dicg_maximum_step_cpu(
        const std::vector<double>& direction,
        const std::vector<double>& x,
        const std::vector<double>& lb,
        const std::vector<double>& ub) const
    {
        double gamma = 1.0;
        for (int e = 0; e < variable_count; ++e) {
            if (direction[e] > 0.0)
                gamma = std::min(gamma, (x[e] - lb[e]) / direction[e]);
            else if (direction[e] < 0.0)
                gamma = std::min(
                    gamma, (ub[e] - x[e]) / -direction[e]);
        }
        return std::max(0.0, gamma);
    }

#ifdef __CUDACC__
    __device__ void compute_v(
        int tid, int n, const double* grad,
        const double* lb, const double* ub, double* v) const
    {
        for (int e = tid; e < n; e += blockDim.x)
            v[e] = 0.0;
        __syncthreads();
        if (n != variable_count)
            return;
        solve_device(tid, grad, lb, ub, v, nullptr);
    }

    __device__ void compute_inface_v(
        int tid, int n, const double* direction, const double* x,
        const double* lb, const double* ub, double* v) const
    {
        for (int e = tid; e < n; e += blockDim.x) v[e] = 0.0;
        __syncthreads();
        if (n != variable_count) return;
        solve_device(tid, direction, lb, ub, v, x);
    }

    __device__ double dicg_maximum_step_local(
        int tid, int n, const double* direction, const double* x,
        const double* lb, const double* ub) const
    {
        double result = 1.0;
        for (int j = tid; j < n; j += blockDim.x) {
            if (direction[j] > 0.0)
                result = fmin(result,
                    (x[j] - lb[j]) / direction[j]);
            else if (direction[j] < 0.0)
                result = fmin(result,
                    (ub[j] - x[j]) / -direction[j]);
        }
        return result;
    }

    __device__ double dicg_sum_maximum_step(
        int, int, const double*, const double*) const
    {
        return 1.0;
    }
#endif

#ifdef __CUDACC__
    struct BlockWorkspace {
        int forced_col[Order], forced_row[Order];
        int rows[Order], cols[Order];
        double u[Order + 1], potentials[Order + 1];
        double minv[Order + 1];
        int p[Order + 1], way[Order + 1];
        unsigned char used[Order + 1];
        double reduction_values[reduction_size];
        int reduction_indices[reduction_size];
        int k, current_col, current_row, next_col;
        double delta;
        int valid, augment_done;
    };

#endif

private:
    static bool solve_cpu(
        const std::vector<double>& cost,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<double>& out,
        const std::vector<double>* face_point)
    {
        if ((int)cost.size() != variable_count
                || (int)lb.size() != variable_count
                || (int)ub.size() != variable_count
                || (face_point
                    && (int)face_point->size() != variable_count))
            return false;

        std::fill(out.begin(), out.end(), 0.0);
        int forced_col[Order];
        int forced_row[Order];
        std::fill(forced_col, forced_col + Order, -1);
        std::fill(forced_row, forced_row + Order, -1);

        for (int r = 0; r < Order; ++r) {
            for (int c = 0; c < Order; ++c) {
                const int e = r * Order + c;
                if (lb[e] > ub[e] + 1e-9)
                    return false;
                const bool face_forced = face_point
                    && (*face_point)[e] >= 1.0 - 1e-9;
                if (lb[e] > 0.5 || face_forced) {
                    if (ub[e] < 0.5 || forced_col[r] >= 0
                            || forced_row[c] >= 0)
                        return false;
                    forced_col[r] = c;
                    forced_row[c] = r;
                    out[e] = 1.0;
                }
            }
        }

        int rows[Order], cols[Order], k = 0, kc = 0;
        for (int i = 0; i < Order; ++i) {
            if (forced_col[i] < 0) rows[k++] = i;
            if (forced_row[i] < 0) cols[kc++] = i;
        }
        if (k != kc)
            return false;
        return hungarian_cpu(cost, ub, rows, cols, k, out, face_point);
    }

    static bool hungarian_cpu(
        const std::vector<double>& cost, const std::vector<double>& ub,
        const int* rows, const int* cols, int k, std::vector<double>& out,
        const std::vector<double>* face_point)
    {
        std::vector<double> u(k + 1), v(k + 1), minv(k + 1);
        std::vector<int> p(k + 1), way(k + 1);
        std::vector<unsigned char> used(k + 1);
        constexpr double inf = 1e300;

        for (int i = 1; i <= k; ++i) {
            p[0] = i;
            std::fill(minv.begin(), minv.end(), inf);
            std::fill(used.begin(), used.end(), 0);
            int j0 = 0;
            do {
                used[j0] = 1;
                const int i0 = p[j0];
                double delta = inf;
                int j1 = -1;
                for (int j = 1; j <= k; ++j) {
                    if (used[j]) continue;
                    const int e = rows[i0 - 1] * Order + cols[j - 1];
                    const bool allowed = ub[e] >= 0.5
                        && (!face_point || (*face_point)[e] > 1e-9);
                    if (allowed) {
                        const double cur = cost[e] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
                if (j1 < 0 || !std::isfinite(delta))
                    return false;
                for (int j = 0; j <= k; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);

            do {
                const int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }

        for (int j = 1; j <= k; ++j)
            out[rows[p[j] - 1] * Order + cols[j - 1]] = 1.0;
        return true;
    }

#ifdef __CUDACC__
    struct DeviceMinPair {
        double value;
        int index;
    };

    __device__ static DeviceMinPair better_pair(
        DeviceMinPair a, DeviceMinPair b)
    {
        if (b.index < 0) return a;
        if (a.index < 0 || b.value < a.value
                || (b.value == a.value && b.index < a.index))
            return b;
        return a;
    }

    __device__ static DeviceMinPair block_min_pair(
        DeviceMinPair value, int tid)
    {
        __shared__ double warp_values[32];
        __shared__ int warp_indices[32];
        const int lane = tid & 31;
        const int warp = tid >> 5;
        const int warp_count = (blockDim.x + 31) >> 5;
        for (int offset = 16; offset > 0; offset >>= 1) {
            DeviceMinPair other{
                __shfl_down_sync(0xffffffff, value.value, offset),
                __shfl_down_sync(0xffffffff, value.index, offset)};
            value = better_pair(value, other);
        }
        if (lane == 0) {
            warp_values[warp] = value.value;
            warp_indices[warp] = value.index;
        }
        __syncthreads();

        DeviceMinPair result{0.0, -1};
        if (warp == 0) {
            result = lane < warp_count
                ? DeviceMinPair{warp_values[lane], warp_indices[lane]}
                : DeviceMinPair{0.0, -1};
            for (int offset = 16; offset > 0; offset >>= 1) {
                DeviceMinPair other{
                    __shfl_down_sync(0xffffffff, result.value, offset),
                    __shfl_down_sync(0xffffffff, result.index, offset)};
                result = better_pair(result, other);
            }
        }
        if (tid == 0) {
            warp_values[0] = result.value;
            warp_indices[0] = result.index;
        }
        __syncthreads();
        return DeviceMinPair{warp_values[0], warp_indices[0]};
    }

    // Block-local shortest-augmenting-path Hungarian algorithm.
    // Threads partition residual columns and cooperatively relax
    // them, reduce minimum slack, and update potentials.
    __device__ static bool solve_device(int, const double* cost, const double* lb,
        const double* ub, double* out, const double* face_point) {
        __shared__ BlockWorkspace workspace;
        LMOBlockContext ctx;
        return solve_cooperative(ctx, workspace, cost, lb, ub, out, face_point);
    }

    template<class Context>
    __device__ static bool solve_cooperative(
        Context& ctx, BlockWorkspace& ws,
        const double* cost, const double* lb,
        const double* ub, double* out, const double* face_point)
    {
        const int tid = ctx.global_tid;
        auto& forced_col = ws.forced_col;
        auto& forced_row = ws.forced_row;
        auto& rows = ws.rows;
        auto& cols = ws.cols;
        auto& u = ws.u;
        auto& potentials = ws.potentials;
        auto& minv = ws.minv;
        auto& p = ws.p;
        auto& way = ws.way;
        auto& used = ws.used;
        auto& reduction_values = ws.reduction_values;
        auto& reduction_indices = ws.reduction_indices;
        auto& k = ws.k;
        auto& current_col = ws.current_col;
        auto& current_row = ws.current_row;
        auto& next_col = ws.next_col;
        auto& delta = ws.delta;
        auto& valid = ws.valid;
        auto& augment_done = ws.augment_done;
        constexpr double inf = 1e300;

        if (tid == 0) {
            valid = 1;
            k = 0;
            for (int i = 0; i < Order; ++i) {
                forced_col[i] = -1;
                forced_row[i] = -1;
            }
            for (int r = 0; r < Order && valid; ++r) {
                for (int c = 0; c < Order; ++c) {
                    const int e = r * Order + c;
                    if (lb[e] > ub[e] + 1e-9) {
                        valid = 0;
                        break;
                    }
                    const bool face_forced = face_point
                        && face_point[e] >= 1.0 - 1e-9;
                    if (lb[e] > 0.5 || face_forced) {
                        if (ub[e] < 0.5 || forced_col[r] >= 0
                                || forced_row[c] >= 0) {
                            valid = 0;
                            break;
                        }
                        forced_col[r] = c;
                        forced_row[c] = r;
                        out[e] = 1.0;
                    }
                }
            }
            int kc = 0;
            for (int i = 0; i < Order; ++i) {
                if (forced_col[i] < 0) rows[k++] = i;
                if (forced_row[i] < 0) cols[kc++] = i;
            }
            if (k != kc) valid = 0;
        }
        ctx.node_sync();
        if (!valid) return false;

        for (int j = tid; j <= k; j += ctx.total_threads) {
            u[j] = 0.0;
            potentials[j] = 0.0;
            p[j] = 0;
        }
        ctx.node_sync();

        for (int i = 1; i <= k; ++i) {
            if (tid == 0) {
                p[0] = i;
                current_col = 0;
                augment_done = 0;
            }
            for (int j = tid; j <= k; j += ctx.total_threads) {
                minv[j] = inf;
                used[j] = 0;
            }
            ctx.node_sync();

            while (!augment_done && valid) {
                if (tid == 0) {
                    used[current_col] = 1;
                    current_row = p[current_col];
                }
                ctx.node_sync();

                for (int j = tid + 1; j <= k; j += ctx.total_threads) {
                    if (used[j]) continue;
                    const int e = rows[current_row - 1] * Order + cols[j - 1];
                    if (ub[e] >= 0.5
                            && (!face_point || face_point[e] > 1e-9)) {
                        const double cur =
                            cost[e] - u[current_row] - potentials[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = current_col;
                        }
                    }
                }
                ctx.node_sync();

                // Fixed-size shared-memory reduction. Padding entries remain
                // invalid, so this is deterministic for every block size and
                // does not rely on shuffle behavior for inactive lanes.
                for (int pos = tid; pos < reduction_size;
                     pos += ctx.total_threads) {
                    const int j = pos + 1;
                    // Unreachable columns have no initialized predecessor.
                    // The finite sentinel must not become an augmenting edge.
                    if (j <= k && !used[j] && minv[j] < inf) {
                        reduction_values[pos] = minv[j];
                        reduction_indices[pos] = j;
                    } else {
                        reduction_values[pos] = inf;
                        reduction_indices[pos] = -1;
                    }
                }
                ctx.node_sync();
                for (int stride = reduction_size >> 1; stride > 0;
                     stride >>= 1) {
                    for (int pos = tid; pos < stride;
                         pos += ctx.total_threads) {
                        const DeviceMinPair left{
                            reduction_values[pos], reduction_indices[pos]};
                        const DeviceMinPair right{
                            reduction_values[pos + stride],
                            reduction_indices[pos + stride]};
                        const DeviceMinPair best = better_pair(left, right);
                        reduction_values[pos] = best.value;
                        reduction_indices[pos] = best.index;
                    }
                    ctx.node_sync();
                }
                if (tid == 0) {
                    delta = reduction_values[0];
                    next_col = reduction_indices[0];
                    if (next_col < 0 || !isfinite(delta)) valid = 0;
                }
                ctx.node_sync();
                if (!valid) break;

                for (int j = tid; j <= k; j += ctx.total_threads) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        potentials[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                ctx.node_sync();
                if (tid == 0) {
                    current_col = next_col;
                    augment_done = p[current_col] == 0;
                }
                ctx.node_sync();
            }
            if (!valid) return false;

            if (tid == 0) {
                do {
                    const int previous_col = way[current_col];
                    p[current_col] = p[previous_col];
                    current_col = previous_col;
                } while (current_col != 0);
            }
            ctx.node_sync();
        }

        for (int j = tid + 1; j <= k; j += ctx.total_threads)
            out[rows[p[j] - 1] * Order + cols[j - 1]] = 1.0;
        ctx.node_sync();
        return true;
    }
#endif
};
