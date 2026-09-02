#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

inline bool dicg_isapprox(double a, double b,
                          double atol = 1e-6, double rtol = 1e-4)
{
    return std::abs(a - b)
        <= atol + rtol * std::max(std::abs(a), std::abs(b));
}

// Host-side DICG primitives for a box.
inline void box_inface_lmo_cpu(
    const std::vector<double>& grad,
    const std::vector<double>& x,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::vector<double>&       a,
    double                     atol = 1e-6,
    double                     rtol = 1e-4)
{
    const int n = (int)grad.size();
    for (int j = 0; j < n; j++) {
        if (dicg_isapprox(x[j], ub[j], atol, rtol))
            a[j] = ub[j];
        else if (dicg_isapprox(x[j], lb[j], atol, rtol))
            a[j] = lb[j];
        else
            a[j] = (grad[j] > 0.0) ? lb[j] : ub[j];
    }
}

inline double box_dicg_maximum_step_cpu(
    const std::vector<double>& direction,
    const std::vector<double>& x,
    const std::vector<double>& lb,
    const std::vector<double>& ub)
{
    double gamma_max = 1.0;
    for (int j = 0; j < (int)x.size(); j++) {
        if (direction[j] < 0.0)
            gamma_max = std::min(
                gamma_max, (ub[j] - x[j]) / -direction[j]);
        else if (direction[j] > 0.0)
            gamma_max = std::min(
                gamma_max, (x[j] - lb[j]) / direction[j]);
    }
    return gamma_max;
}

// Host-side DICG primitives for {x >= 0, sum(x) <= right_side}, with
// branch-and-bound bounds applying to integer variables only.
inline void unit_simplex_inface_lmo_cpu(
    double                           right_side,
    const std::vector<unsigned char>& is_int,
    const std::vector<double>&       grad,
    const std::vector<double>&       x,
    const std::vector<double>&       lb,
    const std::vector<double>&       ub,
    std::vector<double>&             a,
    double                           atol = 1e-6,
    double                           rtol = 1e-4,
    bool                             require_full_mass = false)
{
    const int n = (int)grad.size();
    std::vector<unsigned char> fixed(n, 0), used(n, 0);
    double mass = 0.0;
    const double x_mass = std::accumulate(x.begin(), x.end(), 0.0);
    const bool force_full_mass = require_full_mass
        || dicg_isapprox(x_mass, right_side, atol, rtol);

    for (int j = 0; j < n; j++) {
        a[j] = is_int[j] ? lb[j] : 0.0;
        if (is_int[j]) {
            if (dicg_isapprox(x[j], lb[j], atol, rtol)) {
                fixed[j] = 1;
                a[j] = lb[j];
            } else if (dicg_isapprox(x[j], ub[j], atol, rtol)) {
                fixed[j] = 1;
                a[j] = ub[j];
            }
        } else if (dicg_isapprox(x[j], 0.0, atol, rtol)) {
            fixed[j] = 1;
        }
        mass += a[j];
    }

    if (dicg_isapprox(mass, right_side, atol, rtol))
        return;

    double remaining = right_side - mass;
    while (remaining > 0.0) {
        int best = -1;
        for (int j = 0; j < n; j++) {
            if (fixed[j] || used[j]
                    || (!force_full_mass && grad[j] > 0.0))
                continue;
            if (best < 0 || grad[j] < grad[best]
                    || (grad[j] == grad[best] && j < best))
                best = j;
        }
        if (best < 0)
            break;
        used[best] = 1;

        double add = remaining;
        if (is_int[best])
            add = std::min(std::max(ub[best] - a[best], 0.0),
                           std::floor(remaining));
        a[best] += add;
        remaining = right_side
            - std::accumulate(a.begin(), a.end(), 0.0);
        if (dicg_isapprox(remaining, 0.0, atol, rtol))
            return;
    }
}

inline double unit_simplex_dicg_maximum_step_cpu(
    double                           right_side,
    const std::vector<unsigned char>& is_int,
    const std::vector<double>&       direction,
    const std::vector<double>&       x,
    const std::vector<double>&       lb,
    const std::vector<double>&       ub,
    double                           tol = 1e-6,
    bool                             require_full_mass = false)
{
    double gamma_max = 1.0;
    double sum_x = 0.0;
    double sum_direction = 0.0;
    for (int j = 0; j < (int)x.size(); j++) {
        const double d = direction[j];
        if (d > tol)
            gamma_max = std::min(
                gamma_max, (x[j] - (is_int[j] ? lb[j] : 0.0)) / d);
        else if (d < -tol)
            gamma_max = std::min(
                gamma_max,
                ((is_int[j] ? ub[j] : right_side) - x[j]) / -d);
        if (std::abs(gamma_max) <= tol)
            return 0.0;
        sum_x += x[j];
        sum_direction += d;
    }
    if (require_full_mass && std::abs(sum_direction) > tol)
        return 0.0;
    if (sum_direction < 0.0)
        gamma_max = std::min(
            gamma_max, (sum_x - right_side) / sum_direction);
    return gamma_max;
}

inline bool equality_simplex_linear_feasible_cpu(
    double right_side, const std::vector<double>& v, double tol = 1e-3)
{
    double mass = 0.0;
    for (double value : v) {
        if (value < -tol)
            return false;
        mass += value;
    }
    return std::isfinite(mass) && std::abs(mass - right_side) <= tol;
}

inline bool unit_simplex_linear_feasible_cpu(
    double right_side, const std::vector<double>& v, double tol = 1e-3)
{
    double mass = 0.0;
    for (double value : v) {
        if (value < 0.0)
            return false;
        mass += value;
    }
    return mass <= right_side + tol;
}

inline bool unit_simplex_inface_feasible_cpu(
    double right_side,
    const std::vector<double>& a,
    const std::vector<double>& x,
    double atol = 1e-6,
    double rtol = 1e-4)
{
    const double sum_a = std::accumulate(a.begin(), a.end(), 0.0);
    const double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    if (dicg_isapprox(sum_x, right_side, atol, rtol)
            && !dicg_isapprox(sum_a, right_side, atol, rtol))
        return false;
    return unit_simplex_linear_feasible_cpu(right_side, a);
}
