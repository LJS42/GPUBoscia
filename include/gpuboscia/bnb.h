#pragma once
#include <vector>
#include <cmath>
#include "node.h"

inline bool is_integer_feasible(
    const std::vector<double>& x,
    const std::vector<int>&    int_vars,
    double tol = 1e-6)
{
    for (int i : int_vars) {
        if (i >= (int)x.size()) continue;
        double frac = x[i] - std::floor(x[i]);
        if (frac > tol && frac < 1.0 - tol) return false;
    }
    return true;
}

inline bool is_node_feasible(
    const Node&             node,
    const std::vector<int>& int_vars,
    double tol = 1e-9)
{
    if (node.lb.size() != node.ub.size())
        return false;

    for (size_t j = 0; j < node.lb.size(); j++) {
        if (node.lb[j] > node.ub[j] + tol)
            return false;
    }

    for (int i : int_vars) {
        if (i < 0 || i >= (int)node.lb.size())
            continue;
        const double lo = std::ceil(node.lb[i] - tol);
        const double hi = std::floor(node.ub[i] + tol);
        if (lo > hi)
            return false;
    }

    return true;
}
