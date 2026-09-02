#pragma once
// Algorithm policy interface — include only from .cu files.
//
// An Algorithm struct must provide THREE device methods.  It may set
// static constexpr bool uses_dicg = true to select the two-oracle DICG path;
// if omitted, the solver treats it as false.
//   __device__ double step_size(int global_iter, double gap) const;
//     Step size γ ∈ (0,1].  gap is available for gap-adaptive methods.
//
//   __device__ void update(double* x, const double* v, double gamma, int j) const;
//     In-place FW update for variable j: x[j] += γ·(v[j]−x[j]).
//
//   __device__ bool check_converged(double gap, double epsilon) const;
//     Returns true when the convergence criterion is satisfied.

// ---------------------------------------------------------------
//  Line search strategies
// ---------------------------------------------------------------

__device__ inline double fwls_warp_reduce(double v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ inline double fwls_block_reduce(double v, int tid, int nw, double* ws) {
    v = fwls_warp_reduce(v);
    if ((tid & 31) == 0) ws[tid >> 5] = v;
    __syncthreads();
    double r = (tid < nw) ? ws[tid] : 0.0;
    if (tid < 32) r = fwls_warp_reduce(r);
    __syncthreads();
    return r;
}

template<typename DeviceObjective>
__device__ inline auto fwls_value_after_grad_local(
    const DeviceObjective& obj, int tid, const double*,
    unsigned char* obj_smem, int)
    -> decltype(obj.value_from_factor_local(tid, obj_smem))
{
    return obj.value_from_factor_local(tid, obj_smem);
}

template<typename DeviceObjective>
__device__ inline double fwls_value_after_grad_local(
    const DeviceObjective& obj, int tid, const double* x,
    unsigned char* obj_smem, long)
{
    return obj.value_local(tid, x, obj_smem);
}

// Classic open-loop step: γ = 2/(t+2)
struct OpenLoopLS {
    static constexpr bool uses_secant_line_search = false;
    static constexpr bool scales_dicg_maximum_step = false;
    static constexpr bool safeguards_dicg_drop_step = false;

    __device__ double step(int iter, double /*gap*/) const {
        return 2.0 / (double)(iter + 2);
    }
};

// DICG open-loop schedule expressed as a decaying fraction of the maximum
// feasible step: gamma = gamma_max * 2/(t+2).
struct ScaledOpenLoopLS {
    static constexpr bool uses_secant_line_search = false;
    static constexpr bool scales_dicg_maximum_step = true;
    static constexpr bool safeguards_dicg_drop_step = false;

    __device__ double step(int iter, double /*gap*/) const {
        return 2.0 / (double)(iter + 2);
    }
};

// Scaled open-loop with a drop-step safeguard.  This remains a separate
// policy so the original scaled schedule keeps its exact meaning.
struct SafeguardedScaledOpenLoopLS {
    static constexpr bool uses_secant_line_search = false;
    static constexpr bool scales_dicg_maximum_step = true;
    static constexpr bool safeguards_dicg_drop_step = true;

    __device__ double step(int iter, double /*gap*/) const {
        return 2.0 / (double)(iter + 2);
    }
};

// Exact line search via a per-node secant solve on
//   phi'(gamma) = grad f(x + gamma * dir) · dir,
//   gamma in [0, max_gamma].
//
// This cannot be implemented by step(iter, gap), because it needs x, v, and
// objective gradients at trial points. Device-resident kernels handle it with
// one CUDA block per B&B node. Host-driven solvers should reject this policy
// unless they add the extra batched trial-gradient loop.
struct SecantLS {
    static constexpr bool uses_secant_line_search = true;
    static constexpr bool scales_dicg_maximum_step = false;
    static constexpr bool safeguards_dicg_drop_step = false;

    int    max_iters;
    double tolerance;
    double denom_eps;

    __host__ __device__
    SecantLS(int max_iters_ = 8,
             double tolerance_ = 1e-10,
             double denom_eps_ = 1e-14)
        : max_iters(max_iters_)
        , tolerance(tolerance_)
        , denom_eps(denom_eps_) {}

    __device__ double step(int iter, double /*gap*/) const {
        // Fallback for kernels that have not implemented the secant path.
        return 2.0 / (double)(iter + 2);
    }

    template<typename DeviceObjective>
    __device__ double search(
        DeviceObjective obj,
        int tid,
        int n,
        int nw,
        double gap,
        double max_gamma,
        const double* x,
        const double* dir,
        double* trial_x,
        double* trial_grad,
        double* ws,
        unsigned char* obj_smem,
        double* gamma_s,
        double* phi_lo_s,
        double* phi_hi_s,
        double* lo_s,
        double* hi_s,
        double* f_base_s,
        double* best_f_s,
        double* best_gamma_s,
        bool* line_done_s) const
    {
        const double f_base = fwls_block_reduce(
            obj.value_local(tid, x, obj_smem),
            tid, nw, ws);

        if (tid == 0) {
            *lo_s = 0.0;
            *hi_s = max_gamma;
            *f_base_s = f_base;
            *best_f_s = f_base;
            *best_gamma_s = 0.0;
            *phi_lo_s = -gap;
            *gamma_s = 0.0;
            *line_done_s = (*phi_lo_s >= -tolerance);
        }
        __syncthreads();

        for (int j = tid; j < n; j += blockDim.x)
            trial_x[j] = x[j] + max_gamma * dir[j];
        __syncthreads();

        obj.grad(tid, trial_x, trial_grad, obj_smem);
        __syncthreads();

        double local_phi_hi = 0.0;
        for (int j = tid; j < n; j += blockDim.x)
            local_phi_hi += trial_grad[j] * dir[j];
        const double phi_hi = fwls_block_reduce(local_phi_hi, tid, nw, ws);
        const double f_hi = fwls_block_reduce(
            fwls_value_after_grad_local(
                obj, tid, trial_x, obj_smem, 0),
            tid, nw, ws);
        if (tid == 0) {
            const double f_abs = (*f_base_s >= 0.0) ? *f_base_s : -*f_base_s;
            const double f_allow = tolerance * (1.0 + f_abs);
            *phi_hi_s = phi_hi;
            if (f_hi < *best_f_s ||
                    (*best_gamma_s == 0.0 && f_hi <= *f_base_s + f_allow)) {
                *best_f_s = f_hi;
                *best_gamma_s = max_gamma;
            }
            if (!*line_done_s && *phi_hi_s <= tolerance
                    && f_hi <= *f_base_s + f_allow) {
                *gamma_s = max_gamma;
                *line_done_s = true;
            }
        }
        __syncthreads();

        for (int ls_iter = 0; ls_iter < max_iters; ls_iter++) {
            if (*line_done_s) break;

            if (tid == 0) {
                const double denom = *phi_hi_s - *phi_lo_s;
                double gamma =
                    (denom > denom_eps || denom < -denom_eps)
                        ? *hi_s - *phi_hi_s * (*hi_s - *lo_s) / denom
                        : 0.5 * (*lo_s + *hi_s);
                if (!(gamma > *lo_s && gamma < *hi_s))
                    gamma = 0.5 * (*lo_s + *hi_s);
                *gamma_s = gamma;
            }
            __syncthreads();

            for (int j = tid; j < n; j += blockDim.x)
                trial_x[j] = x[j] + *gamma_s * dir[j];
            __syncthreads();

            obj.grad(tid, trial_x, trial_grad, obj_smem);
            __syncthreads();

            double local_phi = 0.0;
            for (int j = tid; j < n; j += blockDim.x)
                local_phi += trial_grad[j] * dir[j];
            const double phi = fwls_block_reduce(local_phi, tid, nw, ws);
            const double f_trial = fwls_block_reduce(
                fwls_value_after_grad_local(
                    obj, tid, trial_x, obj_smem, 0),
                tid, nw, ws);
            if (tid == 0) {
                const double f_abs = (*f_base_s >= 0.0) ? *f_base_s : -*f_base_s;
                const double f_allow = tolerance * (1.0 + f_abs);
                if (f_trial < *best_f_s ||
                        (*best_gamma_s == 0.0 && f_trial <= *f_base_s + f_allow)) {
                    *best_f_s = f_trial;
                    *best_gamma_s = *gamma_s;
                }
                if (phi < 0.0) {
                    *lo_s = *gamma_s;
                    *phi_lo_s = phi;
                } else {
                    *hi_s = *gamma_s;
                    *phi_hi_s = phi;
                }
                if ((phi <= tolerance && phi >= -tolerance) ||
                    (*hi_s - *lo_s) <= tolerance) {
                    *gamma_s = 0.5 * (*lo_s + *hi_s);
                    *line_done_s = true;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const double f_abs = (*f_base_s >= 0.0) ? *f_base_s : -*f_base_s;
            const double f_allow = tolerance * (1.0 + f_abs);
            if (*best_gamma_s > 0.0 && *best_f_s <= *f_base_s + f_allow)
                *gamma_s = *best_gamma_s;
            else if (!*line_done_s)
                *gamma_s = 0.0;
        }
        __syncthreads();

        return *gamma_s;
    }
};

// ---------------------------------------------------------------
//  Frank-Wolfe with pluggable line search
// ---------------------------------------------------------------
template<typename LS = OpenLoopLS>
struct FrankWolfe {
    static constexpr bool uses_dicg = false;
    static constexpr bool uses_secant_line_search =
        LS::uses_secant_line_search;

    LS ls{};

    __device__ double step_size(int iter, double gap) const {
        return ls.step(iter, gap);
    }
    __device__ void update(double* x, const double* v, double gamma, int j) const {
        x[j] += gamma * (v[j] - x[j]);
    }
    __device__ bool check_converged(double gap, double epsilon) const {
        return gap >= 0.0 && gap < epsilon;
    }
};

using FrankWolfeOpenLoop = FrankWolfe<OpenLoopLS>;
using FrankWolfeSecant   = FrankWolfe<SecantLS>;
using FrankWolfePolicy   = FrankWolfeOpenLoop;  // backward-compatible alias

// Decomposition-Invariant Conditional Gradient.  The LMO supplies a forward
// vertex v, an in-face away vertex a, and the feasible step limit.  The kernel
// passes (a - v) to update().
template<typename LS = OpenLoopLS>
struct DecompositionInvariantFrankWolfe {
    static constexpr bool uses_dicg = true;
    static constexpr bool uses_secant_line_search =
        LS::uses_secant_line_search;
    static constexpr bool scales_dicg_maximum_step =
        LS::scales_dicg_maximum_step;
    static constexpr bool safeguards_dicg_drop_step =
        LS::safeguards_dicg_drop_step;

    LS ls{};

    __device__ double step_size(int iter, double gap) const {
        return ls.step(iter, gap);
    }
    __device__ void update(
        double* x, const double* away_minus_forward,
        double gamma, int j) const
    {
        x[j] -= gamma * away_minus_forward[j];
    }
    __device__ bool check_converged(double gap, double epsilon) const {
        return gap >= 0.0 && gap < epsilon;
    }
};

using DICGOpenLoop = DecompositionInvariantFrankWolfe<OpenLoopLS>;
using DICGScaledOpenLoop =
    DecompositionInvariantFrankWolfe<ScaledOpenLoopLS>;
using DICGSafeguardedScaledOpenLoop =
    DecompositionInvariantFrankWolfe<SafeguardedScaledOpenLoopLS>;
using DICGSecant = DecompositionInvariantFrankWolfe<SecantLS>;
using DICGPolicy   = DICGOpenLoop;
