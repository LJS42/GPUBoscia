// Smallest complete use of gpuboscia::solve: min (x0-.3)^2 + (x1-.7)^2,
// with x0,x1 binary. The unique optimum is [0,1], with objective .18.
#include <gpuboscia/gpuboscia.cuh>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

struct Quadratic {
    static constexpr int n_vars = 2;
    static constexpr int block_threads = 32;
    static constexpr int shared_bytes = 0;
    static constexpr bool supports_quadratic_exact = true;

    __device__ void prepare_block(int, unsigned char*) const {}
    __device__ void grad(int tid, const double* x, double* g, unsigned char*) const {
        if (tid < 2) g[tid] = 2 * (x[tid] - (tid == 0 ? .3 : .7));
    }
    __device__ double value_local(int tid, const double* x, unsigned char*) const {
        if (tid >= 2) return 0;
        const double d = x[tid] - (tid == 0 ? .3 : .7);
        return d * d;
    }
    __device__ double quadratic_curvature_local(
        int tid, const double* direction, unsigned char*) const {
        return tid < 2 ? 2 * direction[tid] * direction[tid] : 0;
    }
};

int main() {
    try {
        Quadratic objective;
        Problem problem{};
        problem.n_vars = 2;
        problem.lb = {0, 0};
        problem.ub = {1, 1};
        problem.int_vars = {0, 1};
        auto settings = gpuboscia::create_default_settings();
        settings.line_search.kind = LineSearchKind::QuadraticExact;
        settings.natural_lower_bound = 0;
        settings.time_limit_seconds = 10;
        const auto result = gpuboscia::solve(problem, objective, BoxLMO{}, settings);
        if (result.x.size() != 2) {
            std::fprintf(stderr, "No incumbent returned.\n");
            return 1;
        }
        std::printf("x=[%.0f, %.0f], objective=%.8f, lower_bound=%.8f\n",
                    result.x[0], result.x[1], result.objective, result.dual_bound);
        return result.x[0] == 0 && result.x[1] == 1
            && std::abs(result.objective - .18) < 1e-8 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
