#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include "node.h"

// ---------------------------------------------------------------
//  Multi-branch strategy type.
//  Takes a parent node, the current relaxation iterate, and the
//  integer variable indices; returns a list of child nodes.
// ---------------------------------------------------------------
using BranchStrategy = std::function<std::vector<Node>(
    const Node&                parent,
    int&                       next_id,
    const std::vector<double>& x,
    const std::vector<int>&    int_vars
)>;

// ---------------------------------------------------------------
//  most_infeasible — returns the single integer variable whose
//  fractional part is closest to 0.5.  Useful as a building block.
// ---------------------------------------------------------------
inline int most_infeasible(
    const std::vector<double>& x,
    const std::vector<int>&    int_vars)
{
    int    best_var  = -1;
    double best_dist = -1.0;
    for (int i : int_vars) {
        double frac = x[i] - std::floor(x[i]);
        double dist = std::min(frac, 1.0 - frac);
        if (dist > best_dist) { best_dist = dist; best_var = i; }
    }
    return best_var;
}

// ---------------------------------------------------------------
//  top_k_infeasible — returns up to k variable indices sorted by
//  descending fractional infeasibility (distance from 0.5).
//  Variables that are already integer (dist ≤ 1e-6) are excluded.
// ---------------------------------------------------------------
inline std::vector<int> top_k_infeasible(
    const std::vector<double>& x,
    const std::vector<int>&    int_vars,
    int                        k,
    double                     integer_tol = 1e-6)
{
    struct Cand { int idx; double dist; };
    std::vector<Cand> cands;
    cands.reserve(int_vars.size());

    for (int i : int_vars) {
        double frac = x[i] - std::floor(x[i]);
        double dist = std::min(frac, 1.0 - frac);
        if (dist > integer_tol) cands.push_back({i, dist});
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b){ return a.dist > b.dist; });

    int n = std::min(k, (int)cands.size());
    std::vector<int> result(n);
    for (int i = 0; i < n; i++) result[i] = cands[i].idx;
    return result;
}

// ---------------------------------------------------------------
//  branch_pow2 — 2^k branching on the k most infeasible variables.
//
//  For each selected variable v, two branches are created:
//    bit = 0  →  ub[v] = floor(x[v])   (round-down / "≤" branch)
//    bit = 1  →  lb[v] = ceil(x[v])    (round-up  / "≥" branch)
//
//  With k=3: 2^3 = 8 children covering all round-down/up
//  combinations of the 3 most infeasible variables.
//
//  If fewer than k variables are fractional, actual_k < k and
//  2^actual_k children are produced.
// ---------------------------------------------------------------
inline std::vector<Node> branch_pow2(
    const Node&                parent,
    int&                       next_id,
    const std::vector<double>& x,
    const std::vector<int>&    int_vars,
    int                        k = 3,
    double                     integer_tol = 1e-6)
{
    auto vars = top_k_infeasible(x, int_vars, k, integer_tol);
    if (vars.empty()) return {};

    const int actual_k     = (int)vars.size();
    const int num_children = 1 << actual_k;

    std::vector<Node> children;
    children.reserve(num_children);

    for (int mask = 0; mask < num_children; mask++) {
        Node child       = parent;
        child.id         = next_id++;
        child.parent_id  = parent.id;
        child.depth      = parent.depth + 1;
        child.dual_bound = parent.dual_bound;
        child.status     = NodeStatus::Active;
        child.relaxation_x.clear();
        child.relaxation_iterations = 0;

        for (int b = 0; b < actual_k; b++) {
            const int    vidx = vars[b];
            const double bval = x[vidx];
            if (mask & (1 << b))
                child.lb[vidx] = std::ceil (bval);   // ≥ ceil(x)
            else
                child.ub[vidx] = std::floor(bval);   // ≤ floor(x)
        }
        children.push_back(child);
    }
    return children;
}
