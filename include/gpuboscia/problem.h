#pragma once
#include <vector>

struct Problem {
    int n_vars;

    std::vector<double> lb;
    std::vector<double> ub;
    std::vector<int>    int_vars;
};
