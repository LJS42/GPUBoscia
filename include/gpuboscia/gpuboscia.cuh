#pragma once

#include <chrono>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "algorithms.cuh"
#include "branching.h"
#include "device_resident_solver.cuh"
#include "line_search_options.h"
#include "node.h"
#include "problem.h"

namespace gpuboscia {

struct Settings {
    DeviceResidentConfig device{};
    FWAlgorithmKind algorithm = FWAlgorithmKind::DICG;
    LineSearchOptions line_search{};
    BranchStrategy branch;
    std::function<void(int, int, double, double, bool)> progress;
    double time_limit_seconds = std::numeric_limits<double>::infinity();
    double natural_lower_bound = -std::numeric_limits<double>::infinity();
};

struct Result {
    double objective = std::numeric_limits<double>::infinity();
    std::vector<double> x;
    double dual_bound = -std::numeric_limits<double>::infinity();
    int nodes_processed = 0;
    double elapsed_seconds = 0.0;
    bool reached_tolerance = false;
    bool reached_time_limit = false;
};

inline Settings create_default_settings()
{
    return Settings{};
}

template<typename LMO, typename = void>
struct lmo_supports_dicg : std::false_type {};

template<typename LMO>
struct lmo_supports_dicg<LMO, std::void_t<decltype(LMO::supports_dicg)>>
    : std::integral_constant<bool, LMO::supports_dicg> {};

template<typename DeviceObjective, typename Algorithm, typename LMO>
Result solve_with_algorithm(
    const Problem& problem,
    DeviceObjective objective,
    LMO lmo,
    const Settings& settings,
    Algorithm algorithm)
{
    BranchStrategy branch = settings.branch;
    if (!branch) {
        const double integer_tol = settings.device.integer_tol;
        branch = [integer_tol](
            const Node& parent, int& next_id, const std::vector<double>& x,
            const std::vector<int>& int_vars) {
                return branch_pow2(
                    parent, next_id, x, int_vars, 3, integer_tol);
            };
    }

    DeviceResidentHybridSolver<DeviceObjective, Algorithm, LMO> solver(
        problem, objective, algorithm, branch, settings.device, lmo,
        settings.time_limit_seconds);
    if (settings.progress)
        solver.set_progress_callback(settings.progress);

    Node root(0, -1, 0, problem.lb, problem.ub);
    if (std::isnan(settings.natural_lower_bound)
            || settings.natural_lower_bound
                == std::numeric_limits<double>::infinity())
        throw std::invalid_argument(
            "natural lower bound must be finite or negative infinity");
    root.dual_bound = settings.natural_lower_bound;
    solver.push(root);

    const auto started = std::chrono::steady_clock::now();
    solver.solve();
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    return Result{
        solver.incumbent_value(), solver.incumbent_x(), solver.dual_bound(),
        solver.nodes_processed(), elapsed, solver.reached_tolerance(),
        solver.reached_time_limit()};
}

// Public solver entry point.  `problem` owns the host objective/gradient used
// for certificate validation and branching; `objective` is its CUDA-callable
// counterpart used by the device-resident relaxation kernel.
template<typename DeviceObjective, typename LMO>
Result solve(
    const Problem& problem,
    DeviceObjective objective,
    LMO lmo,
    const Settings& settings = create_default_settings())
{
    if (settings.line_search.kind == LineSearchKind::QuadraticExact) {
        if constexpr (!SupportsQuadraticExact<DeviceObjective>::value) {
            throw std::invalid_argument(
                "quadratic-exact requires an objective with quadratic curvature support");
        } else {
            if (settings.algorithm == FWAlgorithmKind::DICG) {
                if constexpr (lmo_supports_dicg<LMO>::value)
                    return solve_with_algorithm(problem, objective, lmo, settings,
                                                DICGQuadraticExact{});
                else
                    throw std::invalid_argument("the selected LMO does not support DICG");
            }
            return solve_with_algorithm(problem, objective, lmo, settings,
                                        FrankWolfeQuadraticExact{});
        }
    }
    if (settings.algorithm == FWAlgorithmKind::DICG) {
        if constexpr (!lmo_supports_dicg<LMO>::value) {
            throw std::invalid_argument(
                "the selected LMO does not support DICG; use Frank-Wolfe");
        } else {
            if (settings.line_search.kind == LineSearchKind::Secant)
                return solve_with_algorithm(problem, objective, lmo, settings,
                    DICGSecant{SecantLS{
                        settings.line_search.secant_iters,
                        settings.line_search.secant_tol,
                        settings.line_search.secant_denom_eps}});
            if (settings.line_search.kind == LineSearchKind::ScaledOpenLoop)
                return solve_with_algorithm(
                    problem, objective, lmo, settings, DICGScaledOpenLoop{});
            if (settings.line_search.kind
                    == LineSearchKind::SafeguardedScaledOpenLoop)
                return solve_with_algorithm(problem, objective, lmo, settings,
                    DICGSafeguardedScaledOpenLoop{});
            return solve_with_algorithm(
                problem, objective, lmo, settings, DICGOpenLoop{});
        }
    }

    if (settings.line_search.kind == LineSearchKind::ScaledOpenLoop
            || settings.line_search.kind
                == LineSearchKind::SafeguardedScaledOpenLoop)
        throw std::invalid_argument(
            "scaled open-loop is available only for DICG");
    if (settings.line_search.kind == LineSearchKind::Secant) {
        return solve_with_algorithm(problem, objective, lmo, settings,
            FrankWolfeSecant{SecantLS{
                settings.line_search.secant_iters,
                settings.line_search.secant_tol,
                settings.line_search.secant_denom_eps}});
    }
    return solve_with_algorithm(
        problem, objective, lmo, settings, FrankWolfeOpenLoop{});
}

} // namespace gpuboscia
