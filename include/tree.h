#pragma once
#include <set>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include "node.h"
#include "node_select.h"
#include "branching.h"

class BnBTree {
public:
    double            incumbent_value;
    std::vector<double> incumbent_x;
    int               next_id;
    std::vector<Node> active_nodes;
    NodeSelector      select_node;
    BranchStrategy    select_branch;

    BnBTree(int n_vars, NodeSelector node_sel, BranchStrategy branch_sel)
        : incumbent_value(std::numeric_limits<double>::infinity())
        , incumbent_x(n_vars, 0.0)
        , next_id(1)
        , select_node(std::move(node_sel))
        , select_branch(std::move(branch_sel))
    {}

    void push(const Node& node) {
        active_nodes.push_back(node);
        if (node.dual_bound > -1e300)
            lb_set_.insert(node.dual_bound);
    }

    std::vector<Node> pop_batch(int batch_size) {
        auto indices = select_node(active_nodes, batch_size);
        std::vector<Node> batch;
        batch.reserve(indices.size());
        for (int idx : indices) {
            const Node& nd = active_nodes[idx];
            batch.push_back(nd);
            if (nd.dual_bound > -1e300) {
                auto it = lb_set_.find(nd.dual_bound);
                if (it != lb_set_.end()) lb_set_.erase(it);
            }
        }
        std::sort(indices.rbegin(), indices.rend());
        for (int idx : indices) active_nodes.erase(active_nodes.begin() + idx);
        return batch;
    }

    bool empty()         const { return active_nodes.empty(); }
    int  frontier_size() const { return (int)active_nodes.size(); }

    // Min computed relaxation bound over all open frontier nodes.
    // lb_set_ is kept clean by update_incumbent, so this is never > incumbent_value.
    // Returns incumbent_value when no frontier node has a computed bound.
    double global_lower_bound() const {
        return lb_set_.empty() ? incumbent_value : *lb_set_.begin();
    }

    std::vector<Node> branch_multi(
        const Node&                parent,
        const std::vector<double>& x,
        const std::vector<int>&    int_vars)
    {
        return select_branch(parent, next_id, x, int_vars);
    }

    bool should_prune(const Node& node) const {
        return node.dual_bound >= incumbent_value;
    }

    void update_incumbent(double value, const std::vector<double>& x) {
        if (value < incumbent_value) {
            incumbent_value = value;
            incumbent_x     = x;
            // Eagerly remove all frontier nodes that are now prunable.
            // lb_set_ range erase is O(k log n); active_nodes sweep is O(n).
            lb_set_.erase(lb_set_.lower_bound(value), lb_set_.end());
            active_nodes.erase(
                std::remove_if(active_nodes.begin(), active_nodes.end(),
                    [value](const Node& nd) { return nd.dual_bound >= value; }),
                active_nodes.end());
        }
    }

private:
    // Sorted multiset of dual_bound values for frontier nodes that have a computed
    // relaxation bound (dual_bound > -1e300).  Kept in sync with active_nodes via
    // push/pop/update_incumbent so global_lower_bound() is an O(1) read.
    std::multiset<double> lb_set_;
};
