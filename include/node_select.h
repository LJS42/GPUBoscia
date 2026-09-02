#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include "node.h"

using NodeSelector = std::function<std::vector<int>(
    const std::vector<Node>&, int)>;

inline std::vector<int> best_first(
    const std::vector<Node>& nodes, int batch_size)
{
    std::vector<int> indices(nodes.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    int n = std::min(batch_size, (int)nodes.size());
    std::partial_sort(indices.begin(), indices.begin() + n, indices.end(),
        [&](int a, int b) {
            return nodes[a].dual_bound < nodes[b].dual_bound;
        });
    
    return std::vector<int>(indices.begin(), indices.begin() + n);
}

inline std::vector<int> depth_first(
    const std::vector<Node>& nodes, int batch_size)
{
    std::vector<int> indices(nodes.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    int n = std::min(batch_size, (int)nodes.size());
    std::partial_sort(indices.begin(), indices.begin() + n, indices.end(),
        [&](int a, int b) {
            return nodes[a].depth > nodes[b].depth;
        });
    
    return std::vector<int>(indices.begin(), indices.begin() + n);
}