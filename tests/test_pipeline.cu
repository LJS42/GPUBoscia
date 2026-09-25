// GPU integration tests — requires a CUDA-capable GPU.
//
// Build:  cmake --build build --target test_pipeline
// Run:    ./build/test_pipeline
// Root diagnostic: ./build/test_pipeline --quantization-root 1000
//
// Test problems use a separable quadratic:
//   f(x) = sum_j (x[j] - c[j])^2
// whose optimal integer solution is round(c[j]) clamped to bounds.
// The true optimum is known analytically, so correctness is checkable.

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include <gpuboscia/device_resident_solver.cuh>
#include <gpuboscia/gpuboscia.cuh>
#include <gpuboscia/dense_quadratic_objectives.cuh>
// Reuse the example directly without compiling its executable entry point.
#define GPUBOSCIA_QUANTIZATION_NO_MAIN
#include "../examples/quantization_sparse.cu"
#undef GPUBOSCIA_QUANTIZATION_NO_MAIN

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, msg) \
    do { \
        if (cond) { std::cout << "  PASS: " << (msg) << "\n"; ++g_pass; } \
        else      { std::cerr << "  FAIL: " << (msg) << "\n"; ++g_fail; } \
    } while (0)

#define EXPECT_NEAR(a, b, tol, msg) \
    EXPECT(std::abs((double)(a)-(double)(b)) < (tol), msg)

// ============================================================
//  Helpers
// ============================================================
static BranchStrategy make_branch_fn() {
    return [](const Node& p, int& nid,
              const std::vector<double>& x,
              const std::vector<int>& iv) {
        return branch_pow2(p, nid, x, iv, 3);
    };
}

static double* upload_values(const std::vector<double>& values) {
    double* d;
    cudaMalloc(&d, values.size() * sizeof(double));
    cudaMemcpy(d, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice);
    return d;
}

static std::vector<double> identity_matrix(int n) {
    std::vector<double> Q(n * n, 0.0);
    for (int j = 0; j < n; j++) Q[j*n+j] = 1.0;
    return Q;
}

static double sumsq(const std::vector<double>& x) {
    double v = 0.0;
    for (double a : x) v += a * a;
    return v;
}

template<int N>
__global__ void equality_lmo_probe_kernel(
    EqualitySimplexLMO<N> lmo, int cases,
    const double* grad, const double* neg_grad,
    const double* x, const double* lb, const double* ub,
    double* v, double* a, double* local_steps)
{
    __shared__ double direction[N];
    const int sample = blockIdx.x;
    const int tid = threadIdx.x;
    if (sample >= cases) return;
    const size_t offset = (size_t)sample * N;
    lmo.compute_v(tid, N, grad + offset, lb + offset, ub + offset, v + offset);
    __syncthreads();
    lmo.compute_inface_v(
        tid, N, neg_grad + offset, x + offset,
        lb + offset, ub + offset, a + offset);
    __syncthreads();
    if (tid < N)
        direction[tid] = a[offset + tid] - v[offset + tid];
    __syncthreads();
    if (tid < N) {
        local_steps[offset + tid] = lmo.dicg_maximum_step_local(
            tid, N, direction, x + offset, lb + offset, ub + offset);
    }
}

template<int Order>
__global__ void birkhoff_lmo_probe_kernel(
    BirkhoffLMO<Order> lmo, int cases,
    const double* cost, const double* lb, const double* ub, double* out)
{
    const int sample = blockIdx.x;
    if (sample >= cases) return;
    constexpr int variables = Order * Order;
    const size_t offset = (size_t)sample * variables;
    lmo.compute_v(threadIdx.x, variables, cost + offset,
                  lb + offset, ub + offset, out + offset);
}

template<int Order>
__global__ void birkhoff_dicg_probe_kernel(
    BirkhoffLMO<Order> lmo, int cases,
    const double* direction, const double* neg_direction,
    const double* x, const double* lb, const double* ub,
    double* forward, double* away, double* local_steps)
{
    constexpr int variables = Order * Order;
    __shared__ double pairwise_direction[variables];
    const int sample = blockIdx.x;
    const int tid = threadIdx.x;
    if (sample >= cases) return;
    const size_t offset = (size_t)sample * variables;
    lmo.compute_v(tid, variables, neg_direction + offset,
                  lb + offset, ub + offset, forward + offset);
    __syncthreads();
    lmo.compute_inface_v(tid, variables, direction + offset, x + offset,
                         lb + offset, ub + offset, away + offset);
    __syncthreads();
    for (int e = tid; e < variables; e += blockDim.x)
        pairwise_direction[e] = away[offset + e] - forward[offset + e];
    __syncthreads();
    for (int e = tid; e < variables; e += blockDim.x) {
        local_steps[offset + e] = lmo.dicg_maximum_step_local(
            e, variables, pairwise_direction, x + offset,
            lb + offset, ub + offset);
    }
}

static double exhaustive_assignment_cost(
    int order, const std::vector<double>& cost,
    const std::vector<double>& lb, const std::vector<double>& ub,
    const std::vector<double>* face_point = nullptr)
{
    std::vector<int> permutation(order);
    std::iota(permutation.begin(), permutation.end(), 0);
    double best = std::numeric_limits<double>::infinity();
    do {
        bool feasible = true;
        double value = 0.0;
        for (int r = 0; r < order && feasible; ++r) {
            for (int c = 0; c < order; ++c) {
                const int e = r * order + c;
                const double selected = permutation[r] == c ? 1.0 : 0.0;
                feasible = feasible
                    && selected >= lb[e] - 1e-12
                    && selected <= ub[e] + 1e-12
                    && (!face_point || selected == 0.0
                        || (*face_point)[e] > 1e-9);
            }
            value += cost[r * order + permutation[r]];
        }
        if (feasible) best = std::min(best, value);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}

static void test_birkhoff_lmo_gpu_against_cpu()
{
    std::cout << "\n[BirkhoffLMO block-parallel GPU vs CPU Hungarian]\n";
    constexpr int order = 7;
    constexpr int variables = order * order;
    constexpr int cases = 20;
    std::vector<double> cost(cases * variables);
    std::vector<double> lb(cases * variables, 0.0);
    std::vector<double> ub(cases * variables, 1.0);

    // Every case retains a known cyclic perfect matching. Other arcs are
    // deterministically forbidden, and some matching arcs are forced.
    for (int sample = 0; sample < cases; ++sample) {
        for (int r = 0; r < order; ++r) {
            const int guaranteed_col = (r + sample) % order;
            for (int c = 0; c < order; ++c) {
                const int e = r * order + c;
                const size_t pos = (size_t)sample * variables + e;
                cost[pos] = std::sin(0.37 * (sample + 1) * (e + 2))
                          + 0.013 * e + 0.001 * sample;
                if (c != guaranteed_col
                        && ((r * 11 + c * 7 + sample * 3) % 5 == 0))
                    ub[pos] = 0.0;
            }
        }
        if (sample % 3 != 0) {
            const int forced_row = sample % order;
            const int forced_col = (forced_row + sample) % order;
            lb[(size_t)sample * variables
               + forced_row * order + forced_col] = 1.0;
        }
    }

    double* d_cost = upload_values(cost);
    double* d_lb = upload_values(lb);
    double* d_ub = upload_values(ub);
    double* d_out = nullptr;
    cudaMalloc(&d_out, (size_t)cases * variables * sizeof(double));

    BirkhoffLMO<order> lmo;
    birkhoff_lmo_probe_kernel<order><<<cases, 64>>>(
        lmo, cases, d_cost, d_lb, d_ub, d_out);
    const cudaError_t launch_error = cudaGetLastError();
    EXPECT(launch_error == cudaSuccess,
           "block-parallel Birkhoff probe kernel launched");
    const cudaError_t sync_error = cudaDeviceSynchronize();
    EXPECT(sync_error == cudaSuccess,
           "block-parallel Birkhoff probe kernel completed");

    std::vector<double> gpu(cases * variables);
    cudaMemcpy(gpu.data(), d_out, gpu.size() * sizeof(double),
               cudaMemcpyDeviceToHost);

    bool all_permutations = true;
    bool all_bounds = true;
    bool all_optimal = true;
    for (int sample = 0; sample < cases; ++sample) {
        const size_t offset = (size_t)sample * variables;
        std::vector<double> sample_cost(
            cost.begin() + offset, cost.begin() + offset + variables);
        std::vector<double> sample_lb(
            lb.begin() + offset, lb.begin() + offset + variables);
        std::vector<double> sample_ub(
            ub.begin() + offset, ub.begin() + offset + variables);
        std::vector<double> cpu(variables, 0.0);
        lmo.compute_v_cpu(
            sample_cost, sample_lb, sample_ub, cpu);
        const double exact_value = exhaustive_assignment_cost(
            order, sample_cost, sample_lb, sample_ub);

        double gpu_value = 0.0, cpu_value = 0.0;
        for (int r = 0; r < order; ++r) {
            double row_sum = 0.0, col_sum = 0.0;
            for (int c = 0; c < order; ++c) {
                const int row_edge = r * order + c;
                const int col_edge = c * order + r;
                const double x = gpu[offset + row_edge];
                row_sum += x;
                col_sum += gpu[offset + col_edge];
                all_bounds = all_bounds
                    && x >= sample_lb[row_edge] - 1e-12
                    && x <= sample_ub[row_edge] + 1e-12;
                gpu_value += sample_cost[row_edge] * x;
                cpu_value += sample_cost[row_edge] * cpu[row_edge];
            }
            all_permutations = all_permutations
                && std::abs(row_sum - 1.0) <= 1e-12
                && std::abs(col_sum - 1.0) <= 1e-12;
        }
        const bool optimal = std::abs(gpu_value - exact_value) <= 1e-10
            && std::abs(cpu_value - exact_value) <= 1e-10;
        if (!optimal)
            std::cerr << "  case " << sample << ": GPU=" << gpu_value
                      << " CPU=" << cpu_value
                      << " exhaustive=" << exact_value << "\n";
        all_optimal = all_optimal && optimal;
    }
    EXPECT(all_permutations,
           "GPU outputs one selected edge per row and column");
    EXPECT(all_bounds, "GPU assignments respect forced and forbidden arcs");
    EXPECT(all_optimal, "GPU costs match the exact CPU Hungarian oracle");

    cudaFree(d_cost);
    cudaFree(d_lb);
    cudaFree(d_ub);
    cudaFree(d_out);
}

static void test_birkhoff_dicg_gpu_against_cpu()
{
    std::cout << "\n[BirkhoffLMO DICG minimal-face GPU vs CPU]\n";
    constexpr int order = 7;
    constexpr int variables = order * order;
    constexpr int cases = 12;
    std::vector<double> direction(cases * variables);
    std::vector<double> neg_direction(cases * variables);
    std::vector<double> x(cases * variables, 0.0);
    std::vector<double> lb(cases * variables, 0.0);
    std::vector<double> ub(cases * variables, 1.0);

    for (int sample = 0; sample < cases; ++sample) {
        const int shift_a = sample % order;
        const int shift_b = (shift_a + 1 + sample % (order - 1)) % order;
        const double alpha = sample % 4 == 0 ? 0.0 : 0.37;
        for (int r = 0; r < order; ++r) {
            x[(size_t)sample * variables
              + r * order + (r + shift_a) % order] += 1.0 - alpha;
            x[(size_t)sample * variables
              + r * order + (r + shift_b) % order] += alpha;
            for (int c = 0; c < order; ++c) {
                const int e = r * order + c;
                const size_t pos = (size_t)sample * variables + e;
                direction[pos] = std::cos(0.29 * (sample + 2) * (e + 1))
                               + 0.017 * e;
                neg_direction[pos] = -direction[pos];
            }
        }
    }

    double* d_direction = upload_values(direction);
    double* d_neg_direction = upload_values(neg_direction);
    double* d_x = upload_values(x);
    double* d_lb = upload_values(lb);
    double* d_ub = upload_values(ub);
    double *d_forward = nullptr, *d_away = nullptr, *d_steps = nullptr;
    cudaMalloc(&d_forward, direction.size() * sizeof(double));
    cudaMalloc(&d_away, direction.size() * sizeof(double));
    cudaMalloc(&d_steps, direction.size() * sizeof(double));

    BirkhoffLMO<order> lmo;
    birkhoff_dicg_probe_kernel<order><<<cases, 64>>>(
        lmo, cases, d_direction, d_neg_direction, d_x, d_lb, d_ub,
        d_forward, d_away, d_steps);
    EXPECT(cudaGetLastError() == cudaSuccess,
           "Birkhoff DICG probe kernel launched");
    EXPECT(cudaDeviceSynchronize() == cudaSuccess,
           "Birkhoff DICG probe kernel completed");

    std::vector<double> forward(direction.size()), away(direction.size());
    std::vector<double> steps(direction.size());
    cudaMemcpy(forward.data(), d_forward, forward.size() * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(away.data(), d_away, away.size() * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(steps.data(), d_steps, steps.size() * sizeof(double),
               cudaMemcpyDeviceToHost);

    bool all_in_face = true, all_optimal = true, all_steps = true;
    for (int sample = 0; sample < cases; ++sample) {
        const size_t offset = (size_t)sample * variables;
        std::vector<double> ds(direction.begin() + offset,
                               direction.begin() + offset + variables);
        std::vector<double> xs(x.begin() + offset, x.begin() + offset + variables);
        std::vector<double> lbs(variables, 0.0), ubs(variables, 1.0);
        std::vector<double> cpu_away(variables), cpu_forward(variables);
        lmo.compute_inface_v_cpu(ds, xs, lbs, ubs, cpu_away);
        std::vector<double> nds(variables);
        for (int e = 0; e < variables; ++e) nds[e] = -ds[e];
        lmo.compute_v_cpu(nds, lbs, ubs, cpu_forward);
        const double exact_away = exhaustive_assignment_cost(
            order, ds, lbs, ubs, &xs);

        double gpu_value = 0.0, cpu_value = 0.0, gpu_gamma = 1.0;
        std::vector<double> pairwise(variables);
        for (int e = 0; e < variables; ++e) {
            const double av = away[offset + e];
            all_in_face = all_in_face
                && (xs[e] > 1e-9 || std::abs(av) <= 1e-12)
                && (xs[e] < 1.0 - 1e-9 || std::abs(av - 1.0) <= 1e-12);
            gpu_value += ds[e] * av;
            cpu_value += ds[e] * cpu_away[e];
            // Compare the step formula on the exact direction produced by
            // the GPU. Multiple optimal assignments may differ under ties.
            pairwise[e] = away[offset + e] - forward[offset + e];
            gpu_gamma = std::min(gpu_gamma, steps[offset + e]);
        }
        const bool optimal = std::abs(gpu_value - exact_away) <= 1e-10
            && std::abs(cpu_value - exact_away) <= 1e-10;
        if (!optimal)
            std::cerr << "  DICG case " << sample << ": GPU=" << gpu_value
                      << " CPU=" << cpu_value
                      << " exhaustive=" << exact_away << "\n";
        all_optimal = all_optimal && optimal;
        const double cpu_gamma = lmo.dicg_maximum_step_cpu(
            pairwise, xs, lbs, ubs);
        all_steps = all_steps && std::abs(gpu_gamma - cpu_gamma) <= 1e-10;
    }
    EXPECT(all_in_face, "GPU away vertices stay in the minimal face");
    EXPECT(all_optimal, "GPU in-face costs match CPU Hungarian");
    EXPECT(all_steps, "GPU DICG maximum steps match CPU formula");

    cudaFree(d_direction); cudaFree(d_neg_direction); cudaFree(d_x);
    cudaFree(d_lb); cudaFree(d_ub); cudaFree(d_forward);
    cudaFree(d_away); cudaFree(d_steps);
}

static void enumerate_integer_equality_points_rec(
    int j, int remaining,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::vector<double>& point,
    std::vector<std::vector<double>>& points)
{
    if (j == (int)point.size()) {
        if (remaining == 0) points.push_back(point);
        return;
    }
    const int lo = (int)std::ceil(lb[j]);
    const int hi = (int)std::floor(ub[j]);
    for (int value = lo; value <= hi && value <= remaining; ++value) {
        point[j] = value;
        enumerate_integer_equality_points_rec(
            j + 1, remaining - value, lb, ub, point, points);
    }
}

static double dot_product(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs)
{
    return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), 0.0);
}

static Problem make_problem(int n,
                            const std::vector<double>& c,
                            const std::vector<double>& lb,
                            const std::vector<double>& ub)
{
    Problem p;
    p.n_vars = n;
    p.lb = lb; p.ub = ub;
    p.int_vars.resize(n);
    std::iota(p.int_vars.begin(), p.int_vars.end(), 0);
    p.f = [c, n](const std::vector<double>& x) {
        double f = 0.0;
        for (int j = 0; j < n; j++) { double d = x[j]-c[j]; f += d*d; }
        return f;
    };
    p.grad_f = [c, n](const std::vector<double>& x, std::vector<double>& g) {
        for (int j = 0; j < n; j++) g[j] = 2.0*(x[j]-c[j]);
    };
    return p;
}

static double true_optimal(const std::vector<double>& c,
                            const std::vector<double>& lb,
                            const std::vector<double>& ub)
{
    double total = 0.0;
    for (int j = 0; j < (int)c.size(); j++) {
        double best = std::numeric_limits<double>::infinity();
        for (int k = (int)std::ceil(lb[j]); k <= (int)std::floor(ub[j]); k++) {
            double d = k - c[j];
            best = std::min(best, d * d);
        }
        total += best;
    }
    return total;
}

static void true_simplex_rec(int j,
                             int remaining,
                             const std::vector<double>& c,
                             const std::vector<double>& lb,
                             const std::vector<double>& ub,
                             std::vector<double>& x,
                             double& best)
{
    if (j == (int)c.size()) {
        double f = 0.0;
        for (int k = 0; k < (int)c.size(); k++) {
            const double d = x[k] - c[k];
            f += d * d;
        }
        best = std::min(best, f);
        return;
    }

    const int lo = (int)std::ceil(lb[j]);
    const int hi = std::min((int)std::floor(ub[j]), remaining);
    for (int v = lo; v <= hi; v++) {
        x[j] = (double)v;
        true_simplex_rec(j + 1, remaining - v, c, lb, ub, x, best);
    }
}

static double true_simplex_optimal(const std::vector<double>& c,
                                   const std::vector<double>& lb,
                                   const std::vector<double>& ub,
                                   int rhs)
{
    std::vector<double> x(c.size(), 0.0);
    double best = std::numeric_limits<double>::infinity();
    true_simplex_rec(0, rhs, c, lb, ub, x, best);
    return best;
}

static bool satisfies_simplex(const std::vector<double>& x, double rhs)
{
    double sum = 0.0;
    for (double v : x)
        sum += v;
    return sum <= rhs + 1e-6;
}

static void test_equality_simplex_lmo_host_interface()
{
    std::cout << "\n[EqualitySimplexLMO host interface]\n";
    EqualitySimplexLMO<4> lmo(5.0, 4, {0, 1});
    const std::vector<double> lb = {1.0, 0.0, 0.0, 0.0};
    const std::vector<double> ub = {3.0, 4.0, 5.0, 5.0};
    std::vector<double> v(4, 0.0);
    lmo.compute_v_cpu({4.0, 3.0, 2.0, 1.0}, lb, ub, v);

    const double mass = std::accumulate(v.begin(), v.end(), 0.0);
    EXPECT_NEAR(mass, 5.0, 1e-12,
                "equality LMO fills rhs for all-positive gradient");
    EXPECT_NEAR(v[3], 4.0, 1e-12,
                "equality LMO chooses the smallest-gradient coordinate");
}

static void test_equality_simplex_lmo_against_enumeration()
{
    std::cout << "\n[EqualitySimplexLMO GPU oracle vs exhaustive enumeration]\n";
    constexpr int n = 6;
    constexpr int cases = 12;
    constexpr int rhs = 8;
    const std::vector<double> one_lb = {0, 1, 0, 0, 1, 0};
    const std::vector<double> one_ub = {3, 3, 2, 4, 3, 4};
    std::vector<std::vector<double>> feasible;
    std::vector<double> work(n, 0.0);
    enumerate_integer_equality_points_rec(
        0, rhs, one_lb, one_ub, work, feasible);
    EXPECT(!feasible.empty(), "enumeration produced feasible equality points");

    std::vector<double> grad(cases * n), neg_grad(cases * n);
    std::vector<double> x(cases * n), lb(cases * n), ub(cases * n);
    for (int sample = 0; sample < cases; ++sample) {
        const auto& p = feasible[(sample * 7 + 1) % feasible.size()];
        const auto& q = feasible[(sample * 13 + feasible.size() / 2) % feasible.size()];
        const double alpha = (sample % 3 == 0) ? 0.0
            : (sample % 3 == 1 ? 0.35 : 0.73);
        for (int j = 0; j < n; ++j) {
            const size_t pos = (size_t)sample * n + j;
            grad[pos] = std::sin(0.71 * (sample + 1) * (j + 2))
                      + 0.17 * j - 0.03 * sample;
            neg_grad[pos] = -grad[pos];
            x[pos] = (1.0 - alpha) * p[j] + alpha * q[j];
            lb[pos] = one_lb[j];
            ub[pos] = one_ub[j];
        }
    }

    double* d_grad = upload_values(grad);
    double* d_neg_grad = upload_values(neg_grad);
    double* d_x = upload_values(x);
    double* d_lb = upload_values(lb);
    double* d_ub = upload_values(ub);
    double *d_v = nullptr, *d_a = nullptr, *d_steps = nullptr;
    cudaMalloc(&d_v, cases * n * sizeof(double));
    cudaMalloc(&d_a, cases * n * sizeof(double));
    cudaMalloc(&d_steps, cases * n * sizeof(double));

    EqualitySimplexLMO<n> lmo(rhs, n, {0, 1, 2, 3, 4, 5});
    equality_lmo_probe_kernel<n><<<cases, 32>>>(
        lmo, cases, d_grad, d_neg_grad, d_x, d_lb, d_ub,
        d_v, d_a, d_steps);
    cudaError_t launch_error = cudaGetLastError();
    EXPECT(launch_error == cudaSuccess, "GPU LMO probe kernel launched");
    cudaDeviceSynchronize();

    std::vector<double> v(cases * n), a(cases * n), steps(cases * n);
    cudaMemcpy(v.data(), d_v, v.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(a.data(), d_a, a.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(steps.data(), d_steps, steps.size() * sizeof(double), cudaMemcpyDeviceToHost);

    bool all_forward = true;
    bool all_away = true;
    bool all_feasible = true;
    bool all_steps = true;
    for (int sample = 0; sample < cases; ++sample) {
        std::vector<double> gs(n), xs(n), vs(n), as(n);
        for (int j = 0; j < n; ++j) {
            const size_t pos = (size_t)sample * n + j;
            gs[j] = grad[pos]; xs[j] = x[pos];
            vs[j] = v[pos]; as[j] = a[pos];
        }

        double forward_truth = std::numeric_limits<double>::infinity();
        double away_truth = -std::numeric_limits<double>::infinity();
        for (const auto& point : feasible) {
            forward_truth = std::min(forward_truth, dot_product(gs, point));
            bool same_face = true;
            for (int j = 0; j < n; ++j) {
                if ((std::abs(xs[j] - one_lb[j]) <= 1e-12
                        && point[j] != one_lb[j])
                    || (std::abs(xs[j] - one_ub[j]) <= 1e-12
                        && point[j] != one_ub[j])) {
                    same_face = false;
                    break;
                }
            }
            if (same_face)
                away_truth = std::max(away_truth, dot_product(gs, point));
        }
        all_forward = all_forward
            && std::abs(dot_product(gs, vs) - forward_truth) <= 1e-10;
        all_away = all_away
            && std::abs(dot_product(gs, as) - away_truth) <= 1e-10;

        double sum_v = 0.0, sum_a = 0.0;
        double expected_gamma = 1.0, gpu_gamma = 1.0;
        for (int j = 0; j < n; ++j) {
            sum_v += vs[j]; sum_a += as[j];
            all_feasible = all_feasible
                && vs[j] >= one_lb[j] - 1e-10 && vs[j] <= one_ub[j] + 1e-10
                && as[j] >= one_lb[j] - 1e-10 && as[j] <= one_ub[j] + 1e-10;
            const double direction = as[j] - vs[j];
            if (direction > 1e-6)
                expected_gamma = std::min(expected_gamma,
                    (xs[j] - one_lb[j]) / direction);
            else if (direction < -1e-6)
                expected_gamma = std::min(expected_gamma,
                    (one_ub[j] - xs[j]) / -direction);
            gpu_gamma = std::min(gpu_gamma, steps[(size_t)sample*n+j]);
        }
        all_feasible = all_feasible
            && std::abs(sum_v - rhs) <= 1e-10
            && std::abs(sum_a - rhs) <= 1e-10;
        all_steps = all_steps && std::abs(gpu_gamma - expected_gamma) <= 1e-10;
    }

    EXPECT(all_forward, "GPU forward LMO matches exhaustive minimum");
    EXPECT(all_away, "GPU away LMO matches exhaustive same-face maximum");
    EXPECT(all_feasible, "GPU forward/away vertices satisfy bounds and equality");
    EXPECT(all_steps, "GPU DICG maximum step matches independent formula");

    cudaFree(d_grad); cudaFree(d_neg_grad); cudaFree(d_x);
    cudaFree(d_lb); cudaFree(d_ub); cudaFree(d_v);
    cudaFree(d_a); cudaFree(d_steps);
}

static DeviceResidentConfig make_device_cfg(int batch_size = 16,
                                            int num_streams = 2) {
    DeviceResidentConfig cfg;
    cfg.gpu_batch_size = batch_size;
    cfg.num_streams    = num_streams;
    cfg.gpu_max_iters  = 50000;
    cfg.gpu_eps        = 1e-6;
    cfg.serial_iters   = 500;
    cfg.serial_eps     = 1e-5;
    return cfg;
}

template<int N, typename Algorithm = FrankWolfePolicy>
void run_device_resident_case(const char* name,
                              const std::vector<double>& c,
                              const std::vector<double>& lb,
                              const std::vector<double>& ub,
                              int batch_size,
                              int num_streams,
                              DeviceResidentConfig::DICGOracleParallelism mode =
                                  DeviceResidentConfig::DICGOracleParallelism::Auto)
{
    std::cout << "\n[" << name << " — device-resident global-Q]\n";

    const std::vector<double> Q = identity_matrix(N);
    const double constant = sumsq(c);
    const double f_true = true_optimal(c, lb, ub);

    double* d_Q = upload_values(Q);
    double* d_c = upload_values(c);
    Problem prob = make_problem(N, c, lb, ub);

    using Objective = ILSGlobalCachedObjective<N>;
    Objective obj{d_Q, d_c, constant, Q.data(), c.data()};

    bool threw = false;
    try {
        auto cfg = make_device_cfg(batch_size, num_streams);
        cfg.dicg_oracle_parallelism = mode;
        DeviceResidentHybridSolver<Objective, Algorithm>
            solver(prob, obj, Algorithm{}, make_branch_fn(),
                   cfg);

        Node root(0, -1, 0, lb, ub);
        root.dual_bound = -1e300;
        solver.push(root);
        solver.solve();

        std::cout << "  true f* = " << f_true
                  << "  found = " << solver.incumbent_value() << "\n";
        EXPECT_NEAR(solver.incumbent_value(), f_true, 0.02,
                    "device-resident incumbent within 0.02 of true optimum");
        if (mode == DeviceResidentConfig::DICGOracleParallelism::Paired)
            EXPECT(solver.paired_gpu_batches() > 0,
                   "forced paired DICG launched a paired GPU batch");
    } catch (const std::exception& e) {
        std::cerr << "  EXCEPTION: " << e.what() << "\n";
        threw = true;
    }
    EXPECT(!threw, "device-resident solver completed without exception");

    cudaFree(d_Q);
    cudaFree(d_c);
}

// ============================================================
//  Device-resident global-Q tests.
//  These exercise the user-facing objective API:
//    Objective::f()    on host
//    Objective::grad() on device
// ============================================================
void test_device_resident_n2() {
    const std::vector<double> c  = {1.3, 1.6};
    const std::vector<double> lb = {0.0, 0.0};
    const std::vector<double> ub = {3.0, 3.0};
    run_device_resident_case<2>("Correctness n=2", c, lb, ub, 8, 2);
}

void test_completed_slot_refilled_before_postprocessing() {
    std::cout << "\n[Completed slot immediate refill ordering]\n";

    constexpr int n = 2;
    const std::vector<double> c  = {0.4, 0.6};
    const std::vector<double> lb = {0.0, 0.0};
    const std::vector<double> ub = {1.0, 1.0};
    const std::vector<double> Q = identity_matrix(n);
    const double constant = sumsq(c);

    double* d_Q = upload_values(Q);
    double* d_c = upload_values(c);
    Problem prob = make_problem(n, c, lb, ub);

    using Objective = ILSGlobalCachedObjective<n>;
    Objective obj{d_Q, d_c, constant, Q.data(), c.data()};

    std::vector<DRSSchedulingEvent> events;
    DeviceResidentHybridSolver<Objective, FrankWolfePolicy>
        solver(prob, obj, FrankWolfePolicy{}, make_branch_fn(),
               make_device_cfg(/*batch_size=*/2, /*num_streams=*/1));
    solver.set_scheduling_callback(
        [&](const DRSSchedulingEvent& event) { events.push_back(event); });

    // Two full batches are available before solving starts.  Fixed integral
    // nodes avoid branching, so generation 2 can only be launched from the
    // pre-existing queue.  This makes the ordering assertion deterministic.
    const std::vector<std::vector<double>> fixed_points = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    for (int i = 0; i < (int)fixed_points.size(); ++i) {
        Node node(i, -1, 0, fixed_points[i], fixed_points[i]);
        node.dual_bound = -1e300;
        solver.push(node);
    }
    solver.solve();

    size_t detached_gen1 = events.size();
    size_t dispatched_gen2 = events.size();
    size_t postprocess_gen1 = events.size();
    for (size_t i = 0; i < events.size(); ++i) {
        const auto& event = events[i];
        if (event.slot_id != 0) continue;
        if (event.type == DRSSchedulingEventType::CompletionDetached
                && event.generation == 1)
            detached_gen1 = i;
        if (event.type == DRSSchedulingEventType::Dispatched
                && event.generation == 2)
            dispatched_gen2 = i;
        if (event.type == DRSSchedulingEventType::PostprocessStarted
                && event.generation == 1)
            postprocess_gen1 = i;
    }

    EXPECT(detached_gen1 < events.size(),
           "first completed batch was detached from slot 0");
    EXPECT(dispatched_gen2 < events.size(),
           "slot 0 was refilled with the second full batch");
    EXPECT(postprocess_gen1 < events.size(),
           "CPU post-processing started for the first batch");
    EXPECT(detached_gen1 < dispatched_gen2
               && dispatched_gen2 < postprocess_gen1,
           "completed slot is refilled before its old batch is post-processed");

    cudaFree(d_Q);
    cudaFree(d_c);
}

void test_empty_frontier_branches_before_postprocessing() {
    std::cout << "\n[Empty frontier early branching]\n";
    const std::vector<double> c = {0.5, 0.5};
    const std::vector<double> lb = {0.0, 0.0}, ub = {1.0, 1.0};
    const auto Q = identity_matrix(2);
    double* d_Q = upload_values(Q);
    double* d_c = upload_values(c);
    Problem prob = make_problem(2, c, lb, ub);
    int host_grad_calls = 0;
    const auto original_grad = prob.grad_f;
    prob.grad_f = [&](const std::vector<double>& x, std::vector<double>& g) {
        ++host_grad_calls;
        original_grad(x, g);
    };
    using Objective = ILSGlobalCachedObjective<2>;
    Objective obj{d_Q, d_c, sumsq(c), Q.data(), c.data()};

    // The second case supplies the exact integer lower bound: validation
    // finds an incumbent equal to it and prunes an already expanded root.
    for (double root_bound : {-1e300, 0.5}) {
        std::vector<DRSSchedulingEvent> events;
        int root_branches = 0;
        auto branch = [&](const Node& nd, int& nid,
                          const std::vector<double>& x,
                          const std::vector<int>& iv) {
            if (nd.id == 0) ++root_branches;
            return branch_pow2(nd, nid, x, iv, 3);
        };
        DeviceResidentHybridSolver<Objective, FrankWolfePolicy> solver(
            prob, obj, FrankWolfePolicy{}, branch, make_device_cfg(2, 1));
        solver.set_scheduling_callback(
            [&](const DRSSchedulingEvent& e) { events.push_back(e); });
        Node root(0, -1, 0, lb, ub);
        root.dual_bound = root_bound;
        solver.push(root);
        solver.solve();

        size_t dispatched = events.size(), processed = events.size();
        for (size_t i = 0; i < events.size(); ++i) {
            if (events[i].type == DRSSchedulingEventType::Dispatched
                    && events[i].generation == 2) dispatched = i;
            if (events[i].type == DRSSchedulingEventType::PostprocessStarted
                    && events[i].generation == 1) processed = i;
        }
        EXPECT(dispatched < processed && processed < events.size(),
               "children dispatched before root host post-processing");
        EXPECT(root_branches == 1, "early-expanded root is not branched twice");
        EXPECT_NEAR(solver.incumbent_value(), 0.5, 1e-6,
                    "early branching preserves the integer optimum");
        EXPECT(solver.dual_bound() <= solver.incumbent_value() + 1e-6,
               "overlapping parent and child work preserves a valid bound");
    }
    EXPECT(host_grad_calls == 0, "normal GPU results require no host gradient replay");

    // Input queue order is the reverse of the bounds from the new relaxation.
    std::vector<int> expanded_roots;
    auto ordered_branch = [&](const Node& nd, int& nid,
                              const std::vector<double>& x,
                              const std::vector<int>& iv) {
        if (nd.depth == 0) expanded_roots.push_back(nd.id);
        return branch_pow2(nd, nid, x, iv, 3);
    };
    {
        DeviceResidentHybridSolver<Objective, FrankWolfePolicy> solver(
            prob, obj, FrankWolfePolicy{}, ordered_branch, make_device_cfg(2, 1));
        Node worse(100, -1, 0, {0.75, 0.75}, ub);
        worse.dual_bound = -20.0;
        Node better(200, -1, 0, lb, ub);
        better.dual_bound = -10.0;
        solver.push(worse);
        solver.push(better);
        solver.solve();
        EXPECT(!expanded_roots.empty() && expanded_roots.front() == 200,
               "early branching selects the lowest new GPU bound, not queue order");
        EXPECT_NEAR(solver.incumbent_value(), 0.5, 1e-6,
                    std::string("GPU certificate path preserves optimum with reordered roots, got ")
                        + std::to_string(solver.incumbent_value()));
        EXPECT(is_exact_integer_candidate(solver.incumbent_x(), prob.int_vars),
               "fractional oracle vertex is not stored as the GPU incumbent");
    }
    {
        // Zero steps return the initial integral vertex with a large final
        // gap. It must enter CPU refinement instead of closing the root.
        host_grad_calls = 0;
        auto cfg = make_device_cfg(2, 1);
        cfg.gpu_max_iters = 0;
        DeviceResidentHybridSolver<Objective, FrankWolfePolicy> solver(
            prob, obj, FrankWolfePolicy{}, make_branch_fn(), cfg);
        solver.push(Node(0, -1, 0, lb, ub));
        solver.push(Node(100, -1, 0, {0.75, 0.75}, ub));
        solver.solve();
        EXPECT(host_grad_calls > 0,
               "integral GPU iterate with a large gap triggers CPU refinement");
        EXPECT_NEAR(solver.incumbent_value(), 0.5, 1e-6,
                    "integral unconverged fallback preserves optimum");
        EXPECT(is_exact_integer_candidate(solver.incumbent_x(), prob.int_vars),
               "CPU refinement also rejects fractional oracle incumbents");
    }
    cudaFree(d_Q);
    cudaFree(d_c);
}

void test_device_resident_n4() {
    const std::vector<double> c  = {0.7, 1.6, -0.3, 2.8};
    const std::vector<double> lb = {-1.0, -1.0, -1.0, -1.0};
    const std::vector<double> ub = { 4.0,  4.0,  4.0,  4.0};
    run_device_resident_case<4>("Correctness n=4", c, lb, ub, 16, 2);
}

void test_device_resident_n8() {
    const std::vector<double> c = {0.7, 1.6, -0.3, 2.8, 0.1, -1.4, 3.6, 0.9};
    const std::vector<double> lb(8, -2.0);
    const std::vector<double> ub(8,  4.0);
    run_device_resident_case<8>("Correctness n=8", c, lb, ub, 32, 4);
}

void test_dicg_box_device_resident_n4() {
    const std::vector<double> c  = {0.7, 1.6, -0.3, 2.8};
    const std::vector<double> lb = {-1.0, -1.0, -1.0, -1.0};
    const std::vector<double> ub = { 4.0,  4.0,  4.0,  4.0};
    run_device_resident_case<4, DICGPolicy>(
        "DICG BoxLMO paired correctness n=4", c, lb, ub, 16, 2,
        DeviceResidentConfig::DICGOracleParallelism::Paired);
    run_device_resident_case<4, DICGSafeguardedScaledOpenLoop>(
        "DICG safeguarded scaled open-loop regression n=4",
        c, lb, ub, 16, 2);
    run_device_resident_case<4, DICGSecant>(
        "DICG paired secant regression n=4", c, lb, ub, 16, 2,
        DeviceResidentConfig::DICGOracleParallelism::Paired);
}

void test_simplex_lmo_device_resident_n4() {
    std::cout << "\n[UnitSimplexLMO n=4 — device-resident global-Q]\n";

    constexpr int n = 4;
    constexpr int rhs = 4;
    const std::vector<double> c  = {2.1, 1.8, 0.6, 1.2};
    const std::vector<double> lb = {0.0, 0.0, 0.0, 0.0};
    const std::vector<double> ub = {3.0, 3.0, 2.0, 2.0};
    const std::vector<double> Q = identity_matrix(n);
    const double constant = sumsq(c);
    const double f_true = true_simplex_optimal(c, lb, ub, rhs);

    double* d_Q = upload_values(Q);
    double* d_c = upload_values(c);
    Problem prob = make_problem(n, c, lb, ub);

    using Objective = ILSGlobalCachedObjective<n>;
    Objective obj{d_Q, d_c, constant, Q.data(), c.data()};
    UnitSimplexLMO<n> lmo((double)rhs, n, prob.int_vars);

    DeviceResidentHybridSolver<Objective, FrankWolfePolicy, UnitSimplexLMO<n>>
        solver(prob, obj, FrankWolfePolicy{}, make_branch_fn(),
               make_device_cfg(16, 2), lmo);

    Node root(0, -1, 0, lb, ub); root.dual_bound = -1e300;
    solver.push(root);
    solver.solve();

    std::cout << "  simplex true f* = " << f_true
              << "  found = " << solver.incumbent_value() << "\n";
    EXPECT_NEAR(solver.incumbent_value(), f_true, 0.02,
                "device-resident simplex incumbent within 0.02 of true optimum");
    EXPECT(satisfies_simplex(solver.incumbent_x(), rhs),
           "device-resident simplex incumbent satisfies sum(x) <= rhs");

    auto dicg_cfg = make_device_cfg(16, 2);
    dicg_cfg.dicg_oracle_parallelism =
        DeviceResidentConfig::DICGOracleParallelism::Paired;
    DeviceResidentHybridSolver<Objective, DICGPolicy, UnitSimplexLMO<n>>
        dicg_solver(prob, obj, DICGPolicy{}, make_branch_fn(),
                    dicg_cfg, lmo);
    Node dicg_root(0, -1, 0, lb, ub); dicg_root.dual_bound = -1e300;
    dicg_solver.push(dicg_root);
    dicg_solver.solve();

    std::cout << "  DICG simplex true f* = " << f_true
              << "  found = " << dicg_solver.incumbent_value() << "\n";
    EXPECT_NEAR(dicg_solver.incumbent_value(), f_true, 0.02,
                "DICG simplex incumbent within 0.02 of true optimum");
    EXPECT(satisfies_simplex(dicg_solver.incumbent_x(), rhs),
           "DICG simplex incumbent satisfies sum(x) <= rhs");
    EXPECT(dicg_solver.paired_gpu_batches() > 0,
           "forced paired DICG simplex launched a paired GPU batch");

    cudaFree(d_Q);
    cudaFree(d_c);
}

void test_birkhoff_paired_device_resident() {
    std::cout << "\n[BirkhoffLMO paired device-resident]\n";
    constexpr int order = 4;
    constexpr int n = order * order;
    std::vector<double> target(n, 0.0);
    for (int r = 0; r < order; ++r)
        target[r * order + ((r + 1) % order)] = 1.0;
    const std::vector<double> lb(n, 0.0), ub(n, 1.0);
    const std::vector<double> Q = identity_matrix(n);
    double* d_Q = upload_values(Q);
    double* d_c = upload_values(target);
    Problem prob = make_problem(n, target, lb, ub);
    using Objective = ILSGlobalCachedObjective<n>;
    Objective obj{d_Q, d_c, sumsq(target), Q.data(), target.data()};
    auto cfg = make_device_cfg(4, 1);
    cfg.dicg_oracle_parallelism =
        DeviceResidentConfig::DICGOracleParallelism::Paired;
    BirkhoffLMO<order> lmo;
    DeviceResidentHybridSolver<Objective, DICGPolicy, BirkhoffLMO<order>>
        solver(prob, obj, DICGPolicy{}, make_branch_fn(), cfg, lmo);
    Node root(0, -1, 0, lb, ub);
    root.dual_bound = -1e300;
    solver.push(root);
    solver.solve();
    EXPECT(solver.paired_gpu_batches() > 0,
           "forced paired Birkhoff DICG launched a paired GPU batch");
    EXPECT_NEAR(solver.incumbent_value(), 0.0, 1e-10,
                "paired Birkhoff DICG finds the target permutation");
    cudaFree(d_Q);
    cudaFree(d_c);
}

template<typename Algorithm>
static void run_equality_simplex_convergence_case(
    const char* name,
    Algorithm algorithm,
    const Problem& prob,
    const std::vector<double>& Q,
    const std::vector<double>& Q_transposed,
    const std::vector<double>& linear,
    double constant,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    double rhs,
    int max_iters,
    double eps,
    bool expect_convergence)
{
    constexpr int n = 120;
    double* d_Q = upload_values(Q_transposed);
    double* d_linear = upload_values(linear);
    using Objective = GlobalCachedDenseQuadraticObjective<n, 256>;
    Objective obj{d_Q, d_linear, 1.0, 1.0, constant,
                  Q.data(), linear.data()};
    EqualitySimplexLMO<n> lmo(rhs, n, prob.int_vars);

    DeviceResidentConfig cfg = make_device_cfg(1, 1);
    cfg.gpu_max_iters = max_iters;
    cfg.gpu_eps = eps;
    // This matrix measures line-search convergence, not oracle scheduling.
    // Paired execution has separate forced-path regressions.
    cfg.dicg_oracle_parallelism =
        DeviceResidentConfig::DICGOracleParallelism::Serial;

    BranchStrategy no_branch = [](
        const Node&, int&, const std::vector<double>&,
        const std::vector<int>&) { return std::vector<Node>{}; };

    DeviceResidentHybridSolver<Objective, Algorithm, EqualitySimplexLMO<n>>
        solver(prob, obj, algorithm, no_branch, cfg, lmo);
    Node root(0, -1, 0, lb, ub);
    root.dual_bound = -1e300;
    solver.push(root);
    solver.solve();

    const int iterations = solver.last_gpu_iterations();
    const double gap = solver.last_gpu_gap();
    const bool converged = solver.last_gpu_converged();
    std::cout << "  " << std::left << std::setw(34) << name
              << " iterations=" << std::right << std::setw(6) << iterations
              << "  gap=" << std::scientific << std::setprecision(6) << gap
              << "  converged=" << (converged ? "yes" : "NO") << "\n";

    EXPECT(std::isfinite(gap) && gap >= 0.0,
           std::string(name) + " returns a finite nonnegative FW gap");
    EXPECT(iterations > 0 && iterations <= max_iters,
           std::string(name) + " reports a valid iteration count");
    if (expect_convergence)
        EXPECT(converged,
               std::string(name) + " converges on equality-simplex quadratic");

    cudaFree(d_Q);
    cudaFree(d_linear);
}

static void test_equality_simplex_algorithm_convergence_matrix()
{
    std::cout << "\n[EqualitySimplex convergence matrix — same quadratic]\n";
    std::cout << "  target: main-like 0.5||Ax-b||^2, m=200, n=120, "
                 "sum(x)=200, 0<=x<=8\n";
    constexpr int n = 120;
    constexpr int m = 200;
    constexpr double rhs = 200.0;
    const std::vector<double> lb(n, 0.0);
    const std::vector<double> ub(n, 8.0);
    std::mt19937 rng(0x5eed1234u);
    std::normal_distribution<double> a_dist(
        0.0, 1.0 / std::sqrt(static_cast<double>(m)));
    std::normal_distribution<double> b_dist(0.0, 1.0);
    std::vector<double> A(m * n);
    std::vector<double> b(m);
    for (double& value : A) value = a_dist(rng);
    for (double& value : b) value = b_dist(rng);

    std::vector<double> Q(n * n, 0.0);
    std::vector<double> linear(n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < m; ++i)
                Q[j*n+k] += A[i*n+j] * A[i*n+k];
        for (int i = 0; i < m; ++i)
            linear[j] -= A[i*n+j] * b[i];
    }
    std::vector<double> Q_transposed(n * n);
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
            Q_transposed[k*n+j] = Q[j*n+k];
    double constant = 0.0;
    for (double value : b) constant += 0.5 * value * value;

    Problem prob;
    prob.n_vars = n;
    prob.lb = lb;
    prob.ub = ub;
    prob.int_vars.resize(n);
    std::iota(prob.int_vars.begin(), prob.int_vars.end(), 0);
    prob.f = [Q, linear, constant](const std::vector<double>& x) {
        double value = constant;
        for (int j = 0; j < n; ++j) {
            value += linear[j] * x[j];
            for (int k = 0; k < n; ++k)
                value += 0.5 * x[j] * Q[j*n+k] * x[k];
        }
        return value;
    };
    prob.grad_f = [Q, linear](const std::vector<double>& x,
                              std::vector<double>& grad) {
        for (int j = 0; j < n; ++j) {
            grad[j] = linear[j];
            for (int k = 0; k < n; ++k)
                grad[j] += Q[j*n+k] * x[k];
        }
    };

    constexpr int max_iters = 20000;
    constexpr double eps = 1e-4;

    run_equality_simplex_convergence_case(
        "FW + open-loop", FrankWolfeOpenLoop{},
        prob, Q, Q_transposed, linear, constant,
        lb, ub, rhs, max_iters, eps, false);
    run_equality_simplex_convergence_case(
        "FW + secant", FrankWolfeSecant{SecantLS{16, 1e-10, 1e-14}},
        prob, Q, Q_transposed, linear, constant,
        lb, ub, rhs, max_iters, eps, false);
    run_equality_simplex_convergence_case(
        "DICG + open-loop", DICGOpenLoop{},
        prob, Q, Q_transposed, linear, constant,
        lb, ub, rhs, max_iters, eps, false);
    run_equality_simplex_convergence_case(
        "DICG + scaled open-loop", DICGScaledOpenLoop{},
        prob, Q, Q_transposed, linear, constant,
        lb, ub, rhs, max_iters, eps, false);
    run_equality_simplex_convergence_case(
        "DICG + safeguarded scaled", DICGSafeguardedScaledOpenLoop{},
        prob, Q, Q_transposed, linear, constant,
        lb, ub, rhs, max_iters, eps, true);
    run_equality_simplex_convergence_case(
        "DICG + secant", DICGSecant{SecantLS{16, 1e-10, 1e-14}},
        prob, Q, Q_transposed, linear, constant,
        lb, ub, rhs, max_iters, eps, true);
}

struct SecantToleranceProbeResult {
    int iterations;
    double gap;
    bool converged;
};

static SecantToleranceProbeResult run_secant_tolerance_probe(SecantLS ls)
{
    // At the initial vertex x=1, grad=2e-4 and the FW/pairwise gap is 2e-4.
    // A secant tolerance of 1e-3 therefore returns gamma=0 even though the
    // requested FW tolerance is 1e-6.  The exact minimizer is x=0.5.
    constexpr int n = 1;
    const std::vector<double> Q{4e-4};
    const std::vector<double> linear{-2e-4};
    const std::vector<double> lb{0.0};
    const std::vector<double> ub{1.0};
    double* d_Q = upload_values(Q);
    double* d_linear = upload_values(linear);

    using Objective = GlobalCachedDenseQuadraticObjective<n, 32>;
    Objective objective{
        d_Q, d_linear, 1.0, 1.0, 0.0, Q.data(), linear.data()};

    Problem problem;
    problem.n_vars = n;
    problem.lb = lb;
    problem.ub = ub;
    problem.f = [Q, linear](const std::vector<double>& x) {
        return 0.5 * Q[0] * x[0] * x[0] + linear[0] * x[0];
    };
    problem.grad_f = [Q, linear](const std::vector<double>& x,
                                  std::vector<double>& grad) {
        grad[0] = Q[0] * x[0] + linear[0];
    };

    DeviceResidentConfig cfg = make_device_cfg(1, 1);
    cfg.gpu_max_iters = 64;
    cfg.gpu_eps = 1e-6;
    BranchStrategy no_branch = [](
        const Node&, int&, const std::vector<double>&,
        const std::vector<int>&) { return std::vector<Node>{}; };

    DeviceResidentHybridSolver<Objective, DICGSecant> solver(
        problem, objective, DICGSecant{ls}, no_branch, cfg);
    Node root(0, -1, 0, lb, ub);
    root.dual_bound = -1e300;
    solver.push(root);
    solver.solve();

    const SecantToleranceProbeResult result{
        solver.last_gpu_iterations(),
        solver.last_gpu_gap(),
        solver.last_gpu_converged()};
    cudaFree(d_Q);
    cudaFree(d_linear);
    return result;
}

static void test_secant_tolerance_stall_regression()
{
    std::cout << "\n[Secant tolerance vs FW convergence regression]\n";
    const auto sparse_defaults = run_secant_tolerance_probe(
        SecantLS{8, 1e-3, 1e-4});
    const auto tight = run_secant_tolerance_probe(
        SecantLS{16, 1e-10, 1e-14});

    std::cout << "  sparse defaults: iterations=" << sparse_defaults.iterations
              << " gap=" << std::scientific << sparse_defaults.gap
              << " converged=" << (sparse_defaults.converged ? "yes" : "NO")
              << "\n  tight secant:    iterations=" << tight.iterations
              << " gap=" << tight.gap
              << " converged=" << (tight.converged ? "yes" : "NO") << "\n";

    EXPECT(!sparse_defaults.converged
               && sparse_defaults.iterations == 64
               && sparse_defaults.gap > 1e-6,
           "sparse-Sharpe secant defaults reproduce the zero-step stall");
    EXPECT(tight.converged && tight.iterations < 64 && tight.gap < 1e-6,
           "tight secant tolerance converges on the identical root node");
}


// Opt-in, full-size diagnostic; all root runs restart from the same initial
// vertex. No children are searched, so solver completion is NOT an optimality
// claim. The reported certificate is last_gpu_gap(), not the global B&B gap.
__global__ void quantization_component_probe(
    quantization_sparse::QuantizationObjective objective,
    quantization_sparse::SparseInt4LMO lmo, int component,
    const double* x, double* g, const double* lb, const double* ub,
    double* out, unsigned char* workspace)
{
    if (component == 0) objective.grad(threadIdx.x, x, g, workspace);
    else if (component == 1)
        lmo.compute_v(threadIdx.x, quantization_sparse::Vars, g, lb, ub, out);
    else
        lmo.compute_inface_v(threadIdx.x, quantization_sparse::Vars,
                            g, x, lb, ub, out);
}

static void test_quantization_root_diagnostic(int max_iters, bool exact_only = false)
{
    using namespace quantization_sparse;
    std::cout << "\n[Quantization root diagnostic: 4096 weights, 2000 samples, k=2048, seed=42]\n"
              << "Each row restarts the root; eps=1e-3 is an ABSOLUTE FW gap.\n"
              << "Maximum iteration budget: " << max_iters << ". Stops early if the FW gap meets eps.\n"
              << "No branching; completion does not certify integer optimality.\n" << std::flush;
    Instance data(42, 2000);
    DeviceBuffer coefficients(data.coefficients);
    QuantizationObjective objective{coefficients.p, data.coefficients.data()};
    SparseInt4LMO lmo{Weights/2};
    Problem problem;
    problem.n_vars = Vars;
    problem.lb.assign(Vars, 0); problem.ub.assign(Vars, 1);
    for (int i = 0; i < Weights; ++i) {
        problem.lb[i] = -8; problem.ub[i] = 7;
    }
    problem.int_vars.resize(Vars);
    std::iota(problem.int_vars.begin(), problem.int_vars.end(), 0);
    problem.f = [objective](const std::vector<double>& x) { return objective.f(x); };
    problem.grad_f = [objective](const std::vector<double>& x, std::vector<double>& g) {
        objective.gradient_host(x, g);
    };

    // Use a nontrivial feasible interior point for independent GPU/CPU checks.
    std::vector<double> x(Vars, 0.25), gradient, expected;
    for (int i = 0; i < Weights; ++i) x[i] = 0.25*data.w[i];
    objective.gradient_host(x, gradient);
    DeviceBuffer dx(x), dg(gradient), dlb(problem.lb), dub(problem.ub), output{size_t(Vars)};
    DeviceBuffer workspace{size_t(2*Weights+4)};
    const char* labels[] = {"gradient dH", "forward LMO", "in-face LMO"};
    for (int component = 0; component < 3; ++component) {
        if (component == 2) {
            // The away oracle minimizes the negative gradient.
            for (double& value : gradient) value = -value;
            DRS_CUDA_CHECK(cudaMemcpy(dg.p, gradient.data(), Vars*sizeof(double),
                                      cudaMemcpyHostToDevice));
        }
        auto launch = [&] {
            quantization_component_probe<<<1, QuantizationObjective::block_threads>>>(
                objective, lmo, component, dx.p, dg.p, dlb.p, dub.p, output.p,
                reinterpret_cast<unsigned char*>(workspace.p));
            DRS_CUDA_CHECK(cudaGetLastError());
        };
        launch(); // Warm up before timing.
        DRS_CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        DRS_CUDA_CHECK(cudaEventCreate(&start));
        DRS_CUDA_CHECK(cudaEventCreate(&stop));
        DRS_CUDA_CHECK(cudaEventRecord(start));
        for (int repeat = 0; repeat < 10; ++repeat) launch();
        DRS_CUDA_CHECK(cudaEventRecord(stop));
        DRS_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        DRS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        DRS_CUDA_CHECK(cudaEventDestroy(start));
        DRS_CUDA_CHECK(cudaEventDestroy(stop));
        std::cout << "  isolated " << labels[component] << ": " << ms/10 << " ms/call\n";
        std::vector<double> actual(Vars);
        DRS_CUDA_CHECK(cudaMemcpy(actual.data(), component == 0 ? dg.p : output.p,
                                  Vars*sizeof(double), cudaMemcpyDeviceToHost));
        if (component == 0) expected = gradient;
        else lmo.oracle_cpu(gradient, problem.lb, problem.ub, expected,
                            component == 2 ? x.data() : nullptr);
        double error = 0, scale = 1;
        for (int i = 0; i < Vars; ++i) {
            if (!std::isfinite(actual[i])) error = std::numeric_limits<double>::infinity();
            error = std::max(error, std::abs(actual[i]-expected[i]));
            scale = std::max(scale, std::abs(expected[i]));
        }
        EXPECT(error <= 1e-8*scale, std::string(labels[component])+" agrees with CPU");
        std::cout << std::flush;
    }
    // Check the new GPU Gram construction against the original residual loss.
    const double residual = data.residual(x), gram_loss = objective.f(x);
    EXPECT(std::abs(residual-gram_loss) <= 1e-8*(1+residual),
           "GPU Gram objective agrees with direct X residual");

    BranchStrategy no_branch = [](const Node&, int&, const std::vector<double>&,
                                  const std::vector<int>&) { return std::vector<Node>{}; };
    using Mode = DeviceResidentConfig::DICGOracleParallelism;
    const LineSearchOptions defaults;
    std::vector<int> budgets{1, 10, 100, 1000, 10000, max_iters};
    std::sort(budgets.begin(), budgets.end());
    budgets.erase(std::unique(budgets.begin(), budgets.end()), budgets.end());
    // Default diagnostics include all three searches on identical fresh roots.
    // The exact-only CLI keeps long convergence experiments affordable.
    for (int search : {0, 1, 2}) {
        const bool exact = search == 2;
        const bool tight = search == 1;
        if (exact_only && !exact) continue;
        for (Mode mode : {Mode::Serial, Mode::Auto}) {
            for (int budget : budgets) {
                if (budget > max_iters) continue;
                DeviceResidentConfig cfg;
                cfg.gpu_batch_size = 1; cfg.num_streams = 1;
                cfg.gpu_max_iters = budget;
                cfg.gpu_eps = 1e-3;
                cfg.dicg_oracle_parallelism = mode;
                const SecantLS ls = tight ? SecantLS{16, 1e-10, 1e-14}
                    : SecantLS{defaults.secant_iters, defaults.secant_tol, defaults.secant_denom_eps};
                std::cout << "  RUN line_search=" << (exact ? "quadratic-exact" : tight ? "tight" : "default")
                          << " LMO=" << (mode == Mode::Serial ? "serial" : "auto")
                          << " budget=" << budget << std::flush;
                auto run = [&](auto algorithm) {
                    DeviceResidentHybridSolver<QuantizationObjective, decltype(algorithm), SparseInt4LMO>
                        solver(problem, objective, algorithm, no_branch, cfg, lmo);
                    Node root(0, -1, 0, problem.lb, problem.ub);
                    root.dual_bound = 0;
                    solver.push(root);
                    const auto start = std::chrono::steady_clock::now();
                    solver.solve();
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-start).count();
                    const int iterations = solver.last_gpu_iterations();
                    const double gap = solver.last_gpu_gap();
                    std::cout << " actual_LMO=" << (solver.paired_gpu_batches() ? "paired" : "serial")
                              << " iters=" << iterations << " gap=" << std::scientific << gap
                              << " converged=" << solver.last_gpu_converged()
                              << " wall_s=" << seconds
                              << " wall_ms/iter=" << 1000*seconds/std::max(1, iterations)
                              << std::defaultfloat << '\n' << std::flush;
                    EXPECT(iterations > 0 && iterations <= budget, "root GPU iteration budget respected");
                    EXPECT(std::isfinite(gap) && gap >= -1e-6, "finite root FW certificate");
                    EXPECT(solver.last_gpu_converged() == (gap >= 0 && gap < cfg.gpu_eps),
                           "convergence flag matches absolute FW gap");
                    const auto solution = solver.incumbent_x();
                    EXPECT(lmo.candidate_feasible(solution, problem.lb, problem.ub, 1e-7),
                           "root produces a feasible integer incumbent");
                };
                if (exact) run(DICGQuadraticExact{});
                else run(DICGSecant{ls});
            }
        }
    }
    std::cout << (exact_only
        ? "Interpretation: compare exact-search gap and time across budgets.\n"
        : "Interpretation: compare gap/time across budgets and default/tight/exact search.\n")

              << "Compare actual serial/paired timings to measure LMO pairing benefit.\n"
              << "Component timings are isolated probes, not a full iteration profile;\n"
              << "wall time also includes transfers and CPU postprocessing.\n";
}

// ============================================================
//  main
// ============================================================
// Test the actual device policy with a shifted quadratic, so accidentally
// including the linear term in curvature (or losing the factor two) fails.
__global__ void quadratic_exact_probe(double* output, const double* matrix,
                                      const double* linear) {
    using Objective = GlobalCachedDenseQuadraticObjective<2, 32>;
    __shared__ double direction[2], ws[32], gamma;
    __shared__ __align__(16) unsigned char workspace[Objective::shared_bytes];
    const int tid = threadIdx.x;
    if (tid < 2) direction[tid] = tid == 0 ? 1.0 : -2.0;
    __syncthreads();
    // Hessian = 2*diag(2,3); d^T Hessian d = 28, gap=7 => gamma=1/4.
    Objective obj{matrix, linear, 2.0, 1.0};
    const double step = QuadraticExactLS{}.search(
        obj, tid, 2, 1, 7.0, 1.0, nullptr, direction, nullptr, nullptr,
        ws, workspace, &gamma, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    if (tid == 0) output[0] = step;
}

static void test_quadratic_exact_device() {
    using quantization_sparse::DeviceBuffer;
    DeviceBuffer matrix(std::vector<double>{2,0,0,3});
    DeviceBuffer linear(std::vector<double>{100,-200});
    DeviceBuffer output(size_t(1));
    quadratic_exact_probe<<<1,32>>>(output.p, matrix.p, linear.p);
    DRS_CUDA_CHECK(cudaGetLastError());
    double step = 0;
    DRS_CUDA_CHECK(cudaMemcpy(&step, output.p, sizeof(step), cudaMemcpyDeviceToHost));
    EXPECT(std::abs(step-0.25) < 1e-12, "exact GPU search uses scaled Hessian without linear offset");

    // Interior solution on x0+x1=1: one exact step should reach (0.25,0.75).
    const std::vector<double> Q{1,0,0,1}, target{0.25,0.75};
    DeviceBuffer dQ(Q), dTarget(target);
    using Objective = ILSGlobalCachedObjective<2>;
    Objective objective{dQ.p, dTarget.p, sumsq(target), Q.data(), target.data()};
    Problem problem = make_problem(2, target, {0,0}, {1,1});
    EqualitySimplexLMO<2> lmo(1.0, 2, problem.int_vars);
    BranchStrategy no_branch = [](const Node&, int&, const std::vector<double>&,
                                  const std::vector<int>&) { return std::vector<Node>{}; };
    auto run = [&](auto algorithm, DeviceResidentConfig::DICGOracleParallelism mode) {
        auto cfg = make_device_cfg(1, 1);
        cfg.gpu_max_iters = 1;
        cfg.gpu_eps = 1e-10;
        cfg.dicg_oracle_parallelism = mode;
        DeviceResidentHybridSolver<Objective, decltype(algorithm), EqualitySimplexLMO<2>>
            solver(problem, objective, algorithm, no_branch, cfg, lmo);
        Node root(0, -1, 0, problem.lb, problem.ub);
        root.dual_bound = 0;
        solver.push(root);
        solver.solve();
        EXPECT(solver.last_gpu_iterations() == 1 && solver.last_gpu_converged()
               && std::abs(solver.last_gpu_gap()) < 1e-10,
               "exact search reaches interior quadratic optimum in one GPU iteration");
        if (mode == DeviceResidentConfig::DICGOracleParallelism::Paired)
            EXPECT(solver.paired_gpu_batches() > 0, "exact search exercises paired DICG kernel");
    };
    using Mode = DeviceResidentConfig::DICGOracleParallelism;
    run(FrankWolfeQuadraticExact{}, Mode::Serial);
    run(DICGQuadraticExact{}, Mode::Serial);
    run(DICGQuadraticExact{}, Mode::Paired);

}

static void test_quantization_round_and_prune() {
    using quantization_sparse::SparseInt4LMO;
    std::vector<double> lb{-8,-8,-8,-8,0,0,0,0};
    std::vector<double> ub{7,7,7,7,1,1,1,1};
    const std::vector<double> x{-6.6,1.2,3.6,-0.2,0.1,0.1,0.1,0.1};
    SparseInt4LMO lmo{2};
    auto candidate = lmo.integer_candidate(x, lb, ub);
    EXPECT(candidate == std::vector<double>({-7,0,4,0,1,0,1,0}),
           "round-and-prune keeps largest absolute rounded weights and rebuilds selectors");
    EXPECT(lmo.candidate_feasible(candidate, lb, ub, 0), "round-and-prune candidate is feasible");
    lb[5] = 1; // Small second weight must stay selected.
    candidate = lmo.integer_candidate(x, lb, ub);
    EXPECT(candidate == std::vector<double>({-7,1,0,0,1,1,0,0}),
           "round-and-prune preserves forced-on selector");
    ub[4] = 0; // Largest weight must be off.
    candidate = lmo.integer_candidate(x, lb, ub);
    EXPECT(candidate == std::vector<double>({0,1,4,0,0,1,1,0}),
           "round-and-prune respects forced-off selector");
    lb[5] = 0; ub[4] = 1;
    candidate = SparseInt4LMO{0}.integer_candidate(x, lb, ub);
    EXPECT(candidate == std::vector<double>(8,0), "zero cardinality yields all-zero candidate");
    const std::vector<double> ties{1.2,-1.2,0,0,0,0,0,0};
    candidate = SparseInt4LMO{1}.integer_candidate(ties, lb, ub);
    EXPECT(candidate == std::vector<double>({0,-1,0,0,0,1,0,0}),
           "equal rounded magnitudes use deterministic index tie break");
    lb[4] = lb[5] = 1;
    EXPECT(SparseInt4LMO{1}.integer_candidate(x, lb, ub).empty(),
           "infeasible mandatory cardinality produces no candidate");
    lb[4] = lb[5] = 0; lb[0] = 2.2; ub[0] = 4.8;
    candidate = lmo.integer_candidate(x, lb, ub);
    EXPECT(candidate.size() == 8 && candidate[0] == 3 && candidate[4] == 1
               && lmo.candidate_feasible(candidate, lb, ub, 0),
           "round-and-prune clamps to integer node bounds and never zeros a mandatory weight");
}

int main(int argc, char** argv) {
    bool quantization_diagnostic = true;
    bool exact = false;
    int quantization_iters = 10000;
    try {
        if (argc > 1) {
            if ((std::string(argv[1]) != "--quantization-root" &&
                 std::string(argv[1]) != "--quantization-root-exact") || argc > 3)
                throw std::invalid_argument("Usage: test_pipeline [--quantization-root|--quantization-root-exact [max-iters (default 10000, max 20000)]]");
            quantization_diagnostic = true;
            exact = std::string(argv[1]) == "--quantization-root-exact";
            if (argc == 3) {
                size_t end = 0;
                quantization_iters = std::stoi(argv[2], &end);
                if (argv[2][end] != '\0' || quantization_iters < 1 || quantization_iters > 20000)
                    throw std::invalid_argument("max-iters must be in 1..20000");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    std::cout << "=== GPUBoscia Device-Resident Solver Tests ===\n";
    test_equality_simplex_lmo_host_interface();
    test_quantization_round_and_prune();

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "No CUDA-capable GPU found. Skipping all GPU tests.\n";
        return quantization_diagnostic ? 1 : 0;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";

    try {
        if (quantization_diagnostic) {
            test_quadratic_exact_device();
            test_quantization_root_diagnostic(quantization_iters, exact);
        } else {
            test_equality_simplex_lmo_against_enumeration();
            test_birkhoff_lmo_gpu_against_cpu();
            test_birkhoff_dicg_gpu_against_cpu();
            test_completed_slot_refilled_before_postprocessing();
            test_empty_frontier_branches_before_postprocessing();
            test_device_resident_n2();
            test_device_resident_n4();
            test_device_resident_n8();
            test_dicg_box_device_resident_n4();
            test_simplex_lmo_device_resident_n4();
            test_birkhoff_paired_device_resident();
            test_equality_simplex_algorithm_convergence_matrix();
            test_secant_tolerance_stall_regression();
        }
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception: " << e.what() << "\n";
        ++g_fail;
    }

    std::cout << "\n=== Results: "
              << g_pass << " passed, "
              << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
