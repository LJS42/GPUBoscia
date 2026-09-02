#pragma once

#include <cmath>
#include <algorithm>
#include <limits>

namespace fastbb {

static constexpr double kInf = std::numeric_limits<double>::infinity();

// f(x) - gap is a valid Frank-Wolfe lower bound only for a finite,
// non-negative duality gap.  Invalid certificates must not be clamped to
// zero, because that would incorrectly promote the primal value to a bound.
inline double certified_fw_lower_bound(double inherited_bound,
                                       double primal_value,
                                       double fw_gap) {
    if (!std::isfinite(primal_value) || !std::isfinite(fw_gap)
            || fw_gap < 0.0)
        return inherited_bound;
    return std::max(inherited_bound, primal_value - fw_gap);
}

inline double gap_absolute(double bound, double incumbent) {
    if (incumbent >= kInf || bound <= -kInf) return kInf;
    return std::abs(incumbent - bound);
}

inline double gap_relative(double bound, double incumbent) {
    if (!std::isfinite(bound) || !std::isfinite(incumbent)) return kInf;
    // For minimization, bound is the lower bound L and incumbent is the
    // upper bound U.  Use the symmetric relative-gap definition
    //
    //                 U - L
    //       gaprel = ----------- .
    //                max(|L|,|U|)
    //
    // If L == U == 0, the absolute gap is zero and the relative gap is
    // defined as zero rather than the indeterminate value 0/0.
    const double scale = std::max(std::abs(bound), std::abs(incumbent));
    if (scale == 0.0) return 0.0;
    return (incumbent - bound) / scale;
}

inline bool gap_within_tolerance(double bound,
                                 double incumbent,
                                 double abs_gap_tol,
                                 double rel_gap_tol) {
    if (!std::isfinite(bound) || !std::isfinite(incumbent))
        return false;
    return gap_absolute(bound, incumbent) <= abs_gap_tol
        || gap_relative(bound, incumbent) <= rel_gap_tol;
}

inline double depth_dependent_tolerance(double root_tolerance,
                                        double decay,
                                        double minimum_tolerance,
                                        int depth) {
    if (depth <= 0)
        return std::max(root_tolerance, minimum_tolerance);
    return std::max(minimum_tolerance,
                    root_tolerance * std::pow(decay, depth));
}

} // namespace fastbb
