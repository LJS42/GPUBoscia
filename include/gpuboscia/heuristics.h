#pragma once

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

struct BoxLMO;
template<int MaxN, bool RequireEquality> struct UnitSimplexLMO;

// An oracle vertex need not be integer (e.g. a box with fractional bounds).
// Raw incumbent candidates must be exactly integral on integer coordinates;
// near-integer points go through the rounding/feasibility heuristic instead.
inline bool is_exact_integer_candidate(
    const std::vector<double>& x, const std::vector<int>& int_vars)
{
    for (double v : x)
        if (!std::isfinite(v)) return false;
    for (int i : int_vars)
        if (i < 0 || i >= (int)x.size() || x[i] != std::round(x[i]))
            return false;
    return true;
}

// Round only variables declared integer. Continuous coordinates are preserved.
inline std::vector<double> round_integer_candidate(
    const std::vector<double>& x,
    const std::vector<int>&    int_vars)
{
    std::vector<double> candidate = x;
    for (int i : int_vars)
        if (i >= 0 && i < (int)candidate.size())
            candidate[i] = std::round(candidate[i]);
    return candidate;
}

template<typename LMO, typename = void>
struct HasIntegerCandidateHeuristic : std::false_type {};

template<typename LMO>
struct HasIntegerCandidateHeuristic<LMO, std::void_t<decltype(
    std::declval<const LMO&>().integer_candidate(
        std::declval<const std::vector<double>&>(),
        std::declval<const std::vector<double>&>(),
        std::declval<const std::vector<double>&>()))>> : std::true_type {};

template<typename LMO>
inline std::vector<double> make_integer_candidate(
    const LMO& lmo, const std::vector<double>& x,
    const std::vector<double>& lb, const std::vector<double>& ub,
    const std::vector<int>& int_vars)
{
    if constexpr (HasIntegerCandidateHeuristic<LMO>::value)
        return lmo.integer_candidate(x, lb, ub);
    return round_integer_candidate(x, int_vars);
}

inline bool is_box_point_feasible(
    const std::vector<double>& x,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    double                     tol = 1e-9)
{
    if (x.size() != lb.size() || x.size() != ub.size())
        return false;
    for (size_t j = 0; j < x.size(); j++) {
        if (!std::isfinite(x[j])
                || x[j] < lb[j] - tol || x[j] > ub[j] + tol)
            return false;
    }
    return true;
}

// Feasibility for the bounded simplex {lb <= x <= ub, x >= 0,
// sum(x) <= right_side} used by UnitSimplexLMO.
inline bool is_simplex_point_feasible(
    const std::vector<double>& x,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    double                     right_side,
    double                     tol = 1e-9)
{
    if (!is_box_point_feasible(x, lb, ub, tol))
        return false;
    double mass = 0.0;
    for (double value : x) {
        if (value < -tol)
            return false;
        mass += value;
    }
    return std::isfinite(mass) && mass <= right_side + tol;
}

// Unknown/custom LMOs need their own point-feasibility overload before the
// rounding heuristic can safely be enabled.
template<typename LMO>
inline bool is_rounded_candidate_feasible(
    const LMO& lmo,
    const std::vector<double>& x,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    double tol)
{
    if constexpr (HasIntegerCandidateHeuristic<LMO>::value)
        return lmo.candidate_feasible(x, lb, ub, tol);
    return false;
}

inline bool is_rounded_candidate_feasible(
    const BoxLMO&,
    const std::vector<double>& x,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    double                     tol)
{
    return is_box_point_feasible(x, lb, ub, tol);
}

template<int MaxN, bool RequireEquality>
inline bool is_rounded_candidate_feasible(
    const UnitSimplexLMO<MaxN, RequireEquality>& lmo,
    const std::vector<double>&  x,
    const std::vector<double>&  lb,
    const std::vector<double>&  ub,
    double                      tol)
{
    if (!is_simplex_point_feasible(x, lb, ub, lmo.right_side, tol))
        return false;
    if constexpr (RequireEquality) {
        double mass = 0.0;
        for (double value : x)
            mass += value;
        return std::abs(mass - lmo.right_side) <= tol;
    }
    return true;
}
