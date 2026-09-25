#pragma once
#include <vector>
#include <functional>

struct Problem {
    int n_vars;

    std::function<double(const std::vector<double>&)>                            f;
    std::function<void(const std::vector<double>&, std::vector<double>&)>        grad_f;

    std::vector<double> lb;
    std::vector<double> ub;
    std::vector<int>    int_vars;
};
