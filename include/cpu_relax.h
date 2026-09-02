#pragma once
// Sequential Frank-Wolfe relaxation on CPU.
//
// Used by:
//   Phase 1 (CPU warm-up)  — fast, low-accuracy bound for tree expansion
//   Phase 3 (CPU tail)     — high-accuracy bound for small remaining frontier
//
// Requires Problem::grad_f to be set.

#include <vector>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include "problem.h"
#include "node.h"
#include "gap_utils.h"

struct RelaxResult {
    std::vector<double> x;
    double dual_bound = -1e300;
    double gap        =  1e300;
    bool   converged  = false;

    // Best LMO vertex (extreme point) seen across all FW iterations.
    // Always integer-feasible when box bounds are integers.
    std::vector<double> best_vertex;
    double best_vertex_val = 1e300;
};

// Box LMO:  v[j] = lb[j] if grad[j] > 0 else ub[j]
inline void box_lmo_cpu(
    const std::vector<double>& grad,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::vector<double>&       v)
{
    const int n = (int)grad.size();
    for (int j = 0; j < n; j++)
        v[j] = (grad[j] > 0.0) ? lb[j] : ub[j];
}

template<typename LMO, typename = void>
struct HasComputeVCpu : std::false_type {};

template<typename LMO>
struct HasComputeVCpu<LMO, std::void_t<
    decltype(std::declval<const LMO&>().compute_v_cpu(
        std::declval<const std::vector<double>&>(),
        std::declval<const std::vector<double>&>(),
        std::declval<const std::vector<double>&>(),
        std::declval<std::vector<double>&>()))>> : std::true_type {};

template<typename LMO>
inline void lmo_compute_v_cpu(
    const LMO&                 lmo,
    const std::vector<double>& grad,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::vector<double>&       v)
{
    if constexpr (HasComputeVCpu<LMO>::value)
        lmo.compute_v_cpu(grad, lb, ub, v);
    else
        box_lmo_cpu(grad, lb, ub, v);
}

// A deterministic, non-zero direction used only to obtain a feasible LMO
// vertex before the objective is evaluated.  Any finite direction is valid;
// distinct entries also make tie-breaking deterministic for simple LMOs.
inline std::vector<double> lmo_start_direction(int n)
{
    std::vector<double> direction(n);
    for (int j = 0; j < n; j++)
        direction[j] = -1.0 - static_cast<double>(j) / (n + 1.0);
    return direction;
}

// Runs Frank-Wolfe on the continuous relaxation of a single B&B node.
// Warm-starts from an LMO vertex, since node.lb is only necessarily feasible
// for box constraints.  Returns the best bound and iterate seen.
template<typename LMO>
inline RelaxResult cpu_frank_wolfe(
    const Problem& prob,
    const Node&    node,
    const LMO&     lmo,
    int            max_iters,
    double         epsilon = 1e-6)
{
    const int n = prob.n_vars;
    std::vector<double> x    (n, 0.0);
    std::vector<double> grad (n, 0.0);
    std::vector<double> v    (n, 0.0);

    const std::vector<double> start_direction = lmo_start_direction(n);
    lmo_compute_v_cpu(lmo, start_direction, node.lb, node.ub, x);

    double gap        = 1e300; //inf
    double dual_bound = -1e300; //-inf
    double fx         =  0.0; //inf

    std::vector<double> best_vertex(n);
    double best_vertex_val = 1e300;

    for (int t = 0; t < max_iters; t++) {
        prob.grad_f(x, grad);
        lmo_compute_v_cpu(lmo, grad, node.lb, node.ub, v);

        // v is an LMO extreme point and may provide an incumbent candidate.
        const double fv = prob.f(v);
        if (fv < best_vertex_val) {
            best_vertex_val = fv;
            best_vertex     = v;
        }

        gap = 0.0;
        for (int j = 0; j < n; j++)
            gap += grad[j] * (x[j] - v[j]);

        fx         = prob.f(x);
        dual_bound = fastbb::certified_fw_lower_bound(
            dual_bound, fx, gap);

        if (gap >= 0.0 && gap < epsilon)
            return { x, dual_bound, gap, true, std::move(best_vertex), best_vertex_val };

        const double gamma = 2.0 / (t + 2);
        for (int j = 0; j < n; j++)
            x[j] += gamma * (v[j] - x[j]);
    }

    return { x, dual_bound, gap, false, std::move(best_vertex), best_vertex_val };
}

inline RelaxResult cpu_frank_wolfe(
    const Problem& prob,
    const Node&    node,
    int            max_iters,
    double         epsilon = 1e-6)
{
    struct CpuBoxLMO {
        void compute_v_cpu(
            const std::vector<double>& grad,
            const std::vector<double>& lb,
            const std::vector<double>& ub,
            std::vector<double>&       v) const
        {
            box_lmo_cpu(grad, lb, ub, v);
        }
    };
    return cpu_frank_wolfe(prob, node, CpuBoxLMO{}, max_iters, epsilon);
}
