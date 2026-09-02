// Maximum-weight induced subgraph of fixed cardinality with FastBB.
//
//   minimize  2 sum_{i<j} (1-w_ij) x_i x_j
//   subject to sum_i x_i = Cardinality, x_i in {0,1}.
//
// With fixed cardinality, this is equivalent to maximizing the original
// weights w_ij.  The diagonal convexification below is zero at every binary
// point, so it changes only the continuous relaxation.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "dense_quadratic_objectives.cuh"
#include "fastbb.cuh"
#include "lmo.cuh"
#include "logger.h"

namespace {

constexpr int Vertices = 20;
constexpr int Cardinality = 5;
constexpr int Threads = 32;

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess)                                                  \
        throw std::runtime_error(cudaGetErrorString(error_));                   \
} while (false)

struct Options {
    unsigned long long seed = 222222222;
    double time_limit = 3600.0;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc)
            options.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (arg == "--time-limit" && i + 1 < argc)
            options.time_limit = std::atof(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--seed N] [--time-limit SEC]\n", argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete option: " + arg);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    double* d_Qt = nullptr;
    double* d_linear = nullptr;
    try {
        const Options options = parse_options(argc, argv);

        std::mt19937_64 rng(options.seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::vector<double> weights(Vertices * Vertices, 0.0);
        std::vector<double> complement(Vertices * Vertices, 0.0);
        for (int i = 0; i < Vertices; ++i)
            for (int j = i + 1; j < Vertices; ++j) {
                weights[i * Vertices + j] = weights[j * Vertices + i]
                    = uniform(rng);
                complement[i * Vertices + j]
                    = complement[j * Vertices + i]
                    = 1.0 - weights[i * Vertices + j];
            }

        // f(x) = 2*sum_{i<j} (1-w_ij)*x_i*x_j
        //        + lambda*sum_i x_i*(x_i-1), lambda = Vertices-1.
        // In the 0.5*x'Q*x + linear'*x representation below, Q has diagonal
        // 2*lambda and off-diagonal 2*(1-w_ij).  Since 0 <= 1-w_ij <= 1,
        // Q is symmetric diagonally dominant and positive semidefinite.
        std::vector<double> Q(Vertices * Vertices, 0.0);
        std::vector<double> linear(Vertices, 0.0);
        constexpr double lambda = Vertices - 1.0;
        for (int i = 0; i < Vertices; ++i) {
            for (int j = 0; j < Vertices; ++j) {
                if (i == j) continue;
                Q[i * Vertices + j]
                    = 2.0 * complement[i * Vertices + j];
            }
            Q[i * Vertices + i] = 2.0 * lambda;
            linear[i] = -lambda;
        }

        // The device objective expects Q transposed.  Q happens to be
        // symmetric, but construct the layout explicitly for clarity.
        std::vector<double> Qt(Vertices * Vertices);
        for (int i = 0; i < Vertices; ++i)
            for (int j = 0; j < Vertices; ++j)
                Qt[j * Vertices + i] = Q[i * Vertices + j];

        CUDA_CHECK(cudaMalloc(&d_Qt, Qt.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_linear, linear.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_Qt, Qt.data(), Qt.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_linear, linear.data(),
                              linear.size() * sizeof(double),
                              cudaMemcpyHostToDevice));

        using Objective =
            GlobalCachedDenseQuadraticObjective<Vertices, Threads>;
        Objective objective(d_Qt, d_linear, 1.0, 1.0, 0.0,
                            Q.data(), linear.data());

        Problem problem;
        problem.n_vars = Vertices;
        problem.lb.assign(Vertices, 0.0);
        problem.ub.assign(Vertices, 1.0);
        problem.int_vars.resize(Vertices);
        std::iota(problem.int_vars.begin(), problem.int_vars.end(), 0);
        problem.f = [objective](const std::vector<double>& x) {
            return objective.f(x);
        };
        problem.grad_f = [objective](const std::vector<double>& x,
                                     std::vector<double>& g) {
            for (int i = 0; i < Vertices; ++i) {
                double value = objective.h_linear[i];
                for (int j = 0; j < Vertices; ++j)
                    value += objective.h_Q[i * Vertices + j] * x[j];
                g[i] = value;
            }
        };

        EqualitySimplexLMO<Vertices> lmo(
            static_cast<double>(Cardinality), Vertices, problem.int_vars);

        auto settings = fastbb::create_default_settings();
        settings.algorithm = FWAlgorithmKind::DICG;
        settings.time_limit_seconds = options.time_limit;
        settings.device.gpu_batch_size = 1024;
        settings.device.num_streams = 2;
        settings.device.gpu_max_iters = 1000;
        settings.device.serial_iters = 1000;
        settings.device.abs_gap_tol = 1e-7;
        settings.device.rel_gap_tol = 1e-3;

        // Since sum_i x_i = Cardinality, Cauchy--Schwarz gives
        // sum_i x_i^2 >= Cardinality^2 / Vertices.  The complement-weight
        // term is nonnegative, hence this bounds the continuous relaxation:
        // lambda * sum_i x_i(x_i - 1)
        //   >= lambda * (Cardinality^2 / Vertices - Cardinality).
        constexpr double natural_lower_bound = lambda
            * (static_cast<double>(Cardinality * Cardinality) / Vertices
               - Cardinality);
        settings.natural_lower_bound = natural_lower_bound;

        Logger logger(true, 20000);
        settings.progress = [&logger](int i, int o, double b,
                                    double incumbent, bool fresh) {
            logger.print_row(i, o, b, incumbent, fresh);
        };
        LoggerParams lp;
        lp.traversal_strategy = "best_first (shared queue)";
        lp.branching_strategy = "branch_pow2(k=3, binary)";
        lp.algorithm = fw_algorithm_name(settings.algorithm);
        lp.line_search_method = line_search_name(settings.line_search.kind);
        lp.n_vars = Vertices;
        lp.n_int_vars = static_cast<int>(problem.int_vars.size());
        lp.abs_gap_tol = settings.device.abs_gap_tol;
        lp.rel_gap_tol = settings.device.rel_gap_tol;
        lp.time_limit_s = options.time_limit;
        lp.natural_lower_bound = settings.natural_lower_bound;
        logger.print_header(lp);

        const fastbb::Result result =
            fastbb::solve(problem, objective, lmo, settings);

        SolutionStats stats;
        stats.primal_objective = result.objective;
        stats.dual_bound = result.dual_bound;
        stats.total_nodes = result.nodes_processed;
        stats.total_time_s = result.elapsed_seconds;
        stats.solution = result.x;
        stats.solution_status = result.reached_time_limit ? "Time limit reached"
            : (result.reached_tolerance ? "Reached tolerance" : "Finished");
        logger.print_footer(stats);

        if (result.x.size() != Vertices)
            throw std::runtime_error(
                "FastBB finished without an incumbent solution");

        double total_weight = 0.0;
        double complement_score = 0.0;
        std::vector<int> selected;
        for (int i = 0; i < Vertices; ++i) {
            if (result.x[i] > 0.5) {
                selected.push_back(i);
                std::printf(" %d", i + 1);
            }
        }
        std::printf("\n");
        for (std::size_t a = 0; a < selected.size(); ++a)
            for (std::size_t b = a + 1; b < selected.size(); ++b)
                {
                    total_weight +=
                        weights[selected[a] * Vertices + selected[b]];
                    complement_score +=
                        complement[selected[a] * Vertices + selected[b]];
                }
        std::printf("minimum complement score (without factor 2): %.12f\n",
                    complement_score);
        std::printf("corresponding original internal weight: %.12f\n",
                    total_weight);
        std::printf("FastBB minimization objective: %.12f\n", result.objective);
        std::printf("dual bound: %.12f\n", result.dual_bound);
        std::printf("nodes processed: %d\n", result.nodes_processed);
        std::printf("elapsed seconds: %.6f\n", result.elapsed_seconds);

        CUDA_CHECK(cudaFree(d_linear));
        CUDA_CHECK(cudaFree(d_Qt));
        return 0;
    } catch (const std::exception& error) {
        if (d_linear) cudaFree(d_linear);
        if (d_Qt) cudaFree(d_Qt);
        std::fprintf(stderr, "weighted_clique: %s\n", error.what());
        return 1;
    }
}
