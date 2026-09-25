#pragma once
#include <vector>

enum class NodeStatus {
    Active,   
    Solved    
};

struct Node {
    int id;
    int parent_id;
    int depth;

    std::vector<double> lb;  // lowerbound
    std::vector<double> ub;  // upperbound

    double dual_bound;      
    NodeStatus status;


    Node(int id, int parent_id, int depth,
         std::vector<double> lb, std::vector<double> ub)
        : id(id), parent_id(parent_id), depth(depth),
          lb(std::move(lb)), ub(std::move(ub)),
          dual_bound(-1e300), status(NodeStatus::Active) {}
};
