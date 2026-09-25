#include <gpuboscia/problem.h>
#include <gpuboscia/branching.h>
#include <gpuboscia/gap_utils.h>
#include <gpuboscia/line_search_options.h>
#include <vector>

bool second_translation_unit();

int main() {
    Problem problem{};
    problem.n_vars = 2;
    problem.lb = {0, 0};
    problem.ub = {1, 1};
    problem.int_vars = {0, 1};
    Node root(0, -1, 0, problem.lb, problem.ub);
    int next_id = 1;
    const auto children = branch_pow2(root, next_id, {0.5, 0.5}, problem.int_vars, 1);
    return children.size() == 2 && quadratic_exact_step(2, 4, 1) == 0.5
        && second_translation_unit() ? 0 : 1;
}
