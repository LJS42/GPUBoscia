#include <gpuboscia/device_resident_solver.cuh>
#include <iostream>
#include <vector>
#include <cmath>

#if DRS_HAS_MULTIBLOCK_ATOMICS

__global__ void stress(NodeBarrier* barriers, unsigned* slots, int* errors, int k, unsigned* progress) {
    const int node = blockIdx.x / k, rank = blockIdx.x % k;
    MultiBlockContext ctx(node, rank, k, barriers + node);
    // Slow nodes do not enter their first collective until the fast node
    // has completed 100 generations. A grid-wide phase barrier deadlocks here.
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> ready(*progress);
    if (node && threadIdx.x == 0)
        while (ready.load(cuda::memory_order_acquire) < 100u) {}
    ctx.block_sync();
    // Nodes exit independently, with different amounts of work.
    for (unsigned phase = 1; phase <= 20000u + node * 101u; ++phase) {
        // Every thread publishes a value; representatives must transitively
        // publish writes from their other block threads, not only their own.
        slots[(node * k + rank) * blockDim.x + threadIdx.x] = phase + threadIdx.x;
        ctx.node_sync();
        unsigned sum = 0;
        for (int b = 0; b < k; ++b)
            sum += slots[(node * k + b) * blockDim.x + threadIdx.x];
        if (sum != static_cast<unsigned>(k) * (phase + threadIdx.x)) atomicAdd(errors, 1);
        ctx.node_sync();
        if (node == 0 && rank == 0 && threadIdx.x == 0)
            ready.store(phase, cuda::memory_order_release);
    }
}

// Nonseparable objective with a cross-block sum and multiple internal barriers.
struct SumObjective final : DeviceObjectiveNoPrepare {
    static constexpr int n_vars = 1024, block_threads = 128;
    static constexpr int shared_bytes = sizeof(double);
    static constexpr bool supports_quadratic_exact = true;
    size_t multiblock_workspace_bytes(int k, int) const { return (k + 1) * sizeof(double); }
    __device__ void grad(int tid, const double* x, double* g, unsigned char*) const {
        double sum = 0;
        for (int j = tid; j < n_vars; j += blockDim.x) sum += x[j];
        sum = feasibility_block_sum(sum);
        for (int j = tid; j < n_vars; j += blockDim.x) g[j] = x[j] + sum;
    }
    __device__ void grad_multiblock(MultiBlockContext& ctx, const double* x,
                                    double* g, unsigned char* workspace) const {
        double sum = 0;
        for (int j = ctx.global_tid; j < n_vars; j += ctx.total_threads) sum += x[j];
        sum = feasibility_block_sum(sum);
        auto partial = reinterpret_cast<double*>(workspace);
        if (ctx.thread_rank == 0) partial[ctx.block_rank] = sum;
        ctx.node_sync();
        sum = 0;
        for (int b = 0; b < ctx.block_count; ++b) sum += partial[b];
        for (int j = ctx.global_tid; j < n_vars; j += ctx.total_threads) g[j] = x[j] + sum;
        ctx.node_sync();
    }
    __device__ double quadratic_curvature_local(int tid, const double* d, unsigned char*) const {
        double sum = 0, value = 0;
        for (int j = tid; j < n_vars; j += blockDim.x) sum += d[j];
        sum = feasibility_block_sum(sum);
        for (int j = tid; j < n_vars; j += blockDim.x) value += d[j] * (d[j] + sum);
        return value;
    }
    __device__ double value_local(int tid, const double* x, unsigned char*) const {
        double sum = 0, result = 0;
        for (int j = 0; j < n_vars; ++j) sum += x[j];
        for (int j = tid; j < n_vars; j += blockDim.x) result += .5 * x[j] * (x[j] + sum);
        return result;
    }
};

// A legacy LMO with block collectives must be called by all leader threads.
struct LeaderOnlyBoxLMO : BoxLMO {
    // Hide the inherited hook to exercise legacy block-only fallback.
    void compute_v_multiblock() const = delete;
    int k = 1;
    int* errors = nullptr;
    __device__ void compute_v(int tid, int n, const double* g,
        const double* lb, const double* ub, double* v) const {
        const int participants = __syncthreads_count(true);
        if (tid == 0 && (participants != blockDim.x || blockIdx.x % k != 0))
            atomicAdd(errors, 1);
        BoxLMO::compute_v(tid, n, g, lb, ub, v);
        __syncthreads();
    }
};

// Exercise the optional collective through the same numerical comparisons.
struct CooperativeBoxLMO : BoxLMO {
    int k = 1;
    int* errors = nullptr;
    template<class Context>
    __device__ void compute_v_multiblock(Context& ctx, int n, const double* g,
        const double* lb, const double* ub, double* v) const {
        if (threadIdx.x == 0 && ctx.block_count != k) atomicAdd(errors, 1);
        BoxLMO::compute_v_multiblock(ctx, n, g, lb, ub, v);
    }
};
static_assert(!DRSHasMultiBlockLMO<LeaderOnlyBoxLMO>::value);
static_assert(DRSHasMultiBlockLMO<CooperativeBoxLMO>::value);
static_assert(!DRSHasMultiBlockLMO<BirkhoffLMO<129>>::value);
static_assert(DRSLMOWorkspace<BirkhoffLMO<129>>::bytes == 0);

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void require_near(double value, double reference, double tol, const char* message) {
    require((std::isfinite(value) && std::isfinite(reference) &&
             std::abs(value-reference) <= tol * (1.0 + std::abs(reference))) ||
             (std::isinf(value) && value == reference), message);
}

template<class Function>
void must_throw(Function fn, const char* fragment) {
    try { fn(); }
    catch (const std::exception& e) {
        require(std::string(e.what()).find(fragment) != std::string::npos,
                "incorrect configuration error message");
        return;
    }
    throw std::runtime_error("invalid configuration was accepted");
}

void launch_arithmetic() {
    const size_t scratch = DRSLMOWorkspace<UnitSimplexLMO<17>>::bytes;
    require(scratch % 16 == 0, "LMO scratch stride must be aligned");
    require(drs_multiblock_scratch_offset(3, 1) % 16 == 0,
        "LMO scratch base must be aligned for odd node/block counts");
    require(drs_multiblock_control_bytes(3, 1, scratch)
        == drs_multiblock_scratch_offset(3, 1) + 3*scratch,
        "LMO scratch allocation size mismatch");
    must_throw([] { drs_multiblock_control_bytes(2, 1,
        std::numeric_limits<size_t>::max()); }, "overflows");
    require(drs_multiblock_batch_limit(1000, 4, true, 2, 3, 100) == 1,
            "residency capacity must round down to whole groups");
    require(drs_multiblock_batch_limit(1, 4, true, 2, 3, 100) == 1,
            "requested batch cap not preserved");
    require(drs_multiblock_batch_limit(1000, 2, true, 2, 3, 4) == 2,
            "grid dimension limit not respected");
    must_throw([] { drs_multiblock_batch_limit(1, 0, true, 2, 3, 100); }, "positive");
    must_throw([] { drs_multiblock_batch_limit(1, -4, true, 2, 3, 100); }, "positive");
    must_throw([] { drs_multiblock_batch_limit(1, 8, true, 2, 3, 100); }, "complete node group");
    must_throw([] { drs_multiblock_batch_limit(1, 2, false, 2, 3, 100); }, "cooperative launch");
    must_throw([] { drs_multiblock_batch_limit(1, 2, true, 0, 3, 100); }, "complete node group");
}

template<class Algorithm, class Oracle = LeaderOnlyBoxLMO>
void numerical(int k, int max_iters, double eps, Algorithm alg, bool mixed = false,
               bool heterogeneous = false, bool resume = false) {
    constexpr int n = SumObjective::n_vars, nodes = 2;
    constexpr int vectors = 2 + (DRSUsesDICG<Algorithm>::value ? 2 : 0)
        + (Algorithm::uses_secant_line_search ? 2 : 0);
    SumObjective obj;
    Oracle lmo;
    DRS_CUDA_CHECK(cudaMalloc(&lmo.errors, sizeof(int)));
    DRS_CUDA_CHECK(cudaMemset(lmo.errors, 0, sizeof(int)));
    gpuboscia::RoundingHeuristic heuristic;
    DRSStreamSlot slot;
    size_t stride = ((k + 1) * sizeof(double) + 15u) & ~size_t(15u);
    slot.alloc(nodes, n, vectors, stride);
    DRS_CUDA_CHECK(cudaMalloc(&slot.d_node_controls, drs_multiblock_control_bytes(nodes, k)));
    std::vector<double> bounds(nodes*n, -1.0), tolerances{eps, eps*2};
    if (heterogeneous) tolerances[1] = 1e8;
    if (mixed) bounds[n] = 2.0; // Node one exits before any gradient dispatch.
    DRS_CUDA_CHECK(cudaMemcpy(slot.d_lb, bounds.data(), bounds.size()*sizeof(double), cudaMemcpyHostToDevice));
    std::fill(bounds.begin(), bounds.end(), 1.0);
    DRS_CUDA_CHECK(cudaMemcpy(slot.d_ub, bounds.data(), bounds.size()*sizeof(double), cudaMemcpyHostToDevice));
    DRS_CUDA_CHECK(cudaMemcpy(slot.d_eps, tolerances.data(), nodes*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> reference_x(nodes*n), reference_gap(nodes), reference_value(nodes);
    std::vector<double> reference_best(nodes*n), reference_best_value(nodes);
    std::vector<int> reference_iterations(nodes);
    std::vector<uint8_t> reference_status(nodes);
    int N = nodes;
    double tol = 1e-9;
    uint8_t* flags = nullptr;
    auto kernel = drs_fw_multiblock_kernel<SumObjective, Algorithm, Oracle>;
    int active, device, sms;
    DRS_CUDA_CHECK(cudaGetDevice(&device));
    DRS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 128, 32));
    if (active * sms < nodes*k) {
        slot.free_all();
        DRS_CUDA_CHECK(cudaFree(lmo.errors));
        std::cout << "SKIP numerical K=" << k << ": insufficient capacity for two nodes\n";
        return;
    }
    for (int multi = 0; multi < 2; ++multi) {
        lmo.k = multi ? k : 1;
        DRS_CUDA_CHECK(cudaMemset(slot.d_x, 0, nodes*n*sizeof(double)));
        DRS_CUDA_CHECK(cudaMemset(slot.d_best_vertex, 0, nodes*n*sizeof(double)));
        DRS_CUDA_CHECK(cudaMemset(slot.d_iterations, 0, nodes*sizeof(int)));
        if (resume) {
            std::vector<double> resumed_x(nodes*n, .25);
            const int resumed_iterations[nodes] = {7, 12};
            DRS_CUDA_CHECK(cudaMemcpy(slot.d_x, resumed_x.data(), nodes*n*sizeof(double), cudaMemcpyHostToDevice));
            DRS_CUDA_CHECK(cudaMemcpy(slot.d_iterations, resumed_iterations, nodes*sizeof(int), cudaMemcpyHostToDevice));
        }
        DRS_CUDA_CHECK(cudaMemset(slot.d_node_controls, 0, drs_multiblock_control_bytes(nodes, k)));
        if (!multi) {
            drs_fw_relax_batch_kernel<SumObjective, Algorithm, Oracle>
                <<<nodes, 128, 32>>>(obj, alg, lmo, slot.d_x, slot.d_lb, slot.d_ub,
                    slot.d_gap, slot.d_value, slot.d_iterations, slot.d_cnv,
                    slot.d_best_vertex, slot.d_best_vertex_val, slot.d_workspace,
                    slot.d_objective_workspace, stride, nodes, max_iters, slot.d_eps);
            DRS_CUDA_CHECK(cudaGetLastError());
        } else {
            void* args[] = {&obj, &alg, &lmo, &slot.d_x, &slot.d_lb, &slot.d_ub,
                &slot.d_gap, &slot.d_value, &slot.d_iterations, &slot.d_cnv,
                &slot.d_best_vertex, &slot.d_best_vertex_val, &slot.d_workspace,
                &slot.d_objective_workspace, &stride, &N, &max_iters, &slot.d_eps,
                &flags, &tol, &heuristic, &k, &slot.d_node_controls};
            DRS_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel, dim3(nodes*k), dim3(128), args, 32));
        }
        DRS_CUDA_CHECK(cudaDeviceSynchronize());
        DRS_CUDA_CHECK(cudaMemcpy(slot.h_x, slot.d_x, nodes*n*sizeof(double), cudaMemcpyDeviceToHost));
        DRS_CUDA_CHECK(cudaMemcpy(slot.h_gap, slot.d_gap, nodes*sizeof(double), cudaMemcpyDeviceToHost));
        DRS_CUDA_CHECK(cudaMemcpy(slot.h_value, slot.d_value, nodes*sizeof(double), cudaMemcpyDeviceToHost));
        DRS_CUDA_CHECK(cudaMemcpy(slot.h_iterations, slot.d_iterations, nodes*sizeof(int), cudaMemcpyDeviceToHost));
        DRS_CUDA_CHECK(cudaMemcpy(slot.h_cnv, slot.d_cnv, nodes*sizeof(uint8_t), cudaMemcpyDeviceToHost));
        DRS_CUDA_CHECK(cudaMemcpy(slot.h_best_vertex, slot.d_best_vertex, nodes*n*sizeof(double), cudaMemcpyDeviceToHost));
        DRS_CUDA_CHECK(cudaMemcpy(slot.h_best_vertex_val, slot.d_best_vertex_val, nodes*sizeof(double), cudaMemcpyDeviceToHost));
        if (!multi) {
            std::copy_n(slot.h_x, nodes*n, reference_x.begin());
            std::copy_n(slot.h_gap, nodes, reference_gap.begin());
            std::copy_n(slot.h_value, nodes, reference_value.begin());
            std::copy_n(slot.h_iterations, nodes, reference_iterations.begin());
            std::copy_n(slot.h_cnv, nodes, reference_status.begin());
            std::copy_n(slot.h_best_vertex, nodes*n, reference_best.begin());
            std::copy_n(slot.h_best_vertex_val, nodes, reference_best_value.begin());
        } else {
            for (int j = 0; j < nodes*n; ++j)
                require_near(slot.h_x[j], reference_x[j], 1e-10, "iterate mismatch");
            for (int j = 0; j < nodes; ++j) {
                require_near(slot.h_gap[j], reference_gap[j], 1e-9, "gap mismatch");
                require_near(slot.h_value[j], reference_value[j], 1e-9, "value mismatch");
                require_near(slot.h_best_vertex_val[j], reference_best_value[j], 1e-9, "best candidate value mismatch");
                if (slot.h_best_vertex_val[j] < 1e299)
                    for (int t = 0; t < n; ++t)
                        require_near(slot.h_best_vertex[j*n+t], reference_best[j*n+t], 1e-10,
                                     "best candidate vector mismatch");
                require(slot.h_iterations[j] == reference_iterations[j] &&
                        slot.h_cnv[j] == reference_status[j], "iteration/status mismatch");
            }
            if (heterogeneous)
                require(slot.h_iterations[0] > slot.h_iterations[1] && slot.h_iterations[1] == 1,
                        "different FW iteration counts not exercised");
            if (resume)
                require(slot.h_iterations[0] == 7+max_iters && slot.h_iterations[1] == 12+max_iters,
                        "iteration continuation not preserved");
            if (mixed) require(slot.h_cnv[1] == DRSInfeasible && slot.h_iterations[1] == 0,
                               "infeasible helper group did not exit uniformly");
            if (eps > 1e6) require(slot.h_iterations[0] == 1 && slot.h_cnv[0] == DRSIntegralConverged,
                                  "early convergence not exercised");
            if (max_iters == 1 && eps < 1e-6)
                require(slot.h_iterations[0] == 1 && slot.h_cnv[0] == DRSIntegralUnconverged,
                        "max-iteration termination not exercised");
        }
    }
    int errors;
    DRS_CUDA_CHECK(cudaMemcpy(&errors, lmo.errors, sizeof(int), cudaMemcpyDeviceToHost));
    require(errors == 0, "LMO executed outside complete leader block");
    DRS_CUDA_CHECK(cudaFree(lmo.errors));
    slot.free_all();
}

struct LegacyObjective : DeviceObjectiveNoPrepare {
    static constexpr int n_vars = SumObjective::n_vars;
    static constexpr int block_threads = SumObjective::block_threads;
    static constexpr int shared_bytes = SumObjective::shared_bytes;
    __device__ void grad(int tid, const double* x, double* g, unsigned char* w) const {
        SumObjective{}.grad(tid, x, g, w);
    }
    __device__ double value_local(int tid, const double* x, unsigned char* w) const {
        return SumObjective{}.value_local(tid, x, w);
    }
};

struct OversizedWorkspaceObjective : LegacyObjective {
    size_t multiblock_workspace_bytes(int, int) const { return std::numeric_limits<size_t>::max(); }
    __device__ void grad_multiblock(MultiBlockContext& ctx, const double* x,
                                   double* g, unsigned char* w) const {
        SumObjective{}.grad_multiblock(ctx, x, g, w);
    }
};

void solver_integration(int k, bool cooperative) {
    Problem problem{SumObjective::n_vars,
        std::vector<double>(SumObjective::n_vars, -1),
        std::vector<double>(SumObjective::n_vars, 1), {}};
    BranchStrategy branch = [](const Node&, int&, const std::vector<double>&,
                               const std::vector<int>&) { return std::vector<Node>{}; };
    DeviceResidentConfig cfg;
    cfg.execution_mode = FWExecutionMode::MultiBlock;
    cfg.blocks_per_node = k;
    cfg.gpu_batch_size = 100000;
    cfg.num_streams = 4;
    cfg.gpu_max_iters = 20;
    cfg.gpu_eps = cfg.gpu_eps_min = 1e-8;
    auto construct = [&](DeviceResidentConfig invalid) {
        DeviceResidentHybridSolver<SumObjective, FrankWolfeSecant> solver(
            problem, {}, {}, branch, invalid);
    };
    auto invalid = cfg;
    invalid.blocks_per_node = 0;
    must_throw([&] { construct(invalid); }, "positive");
    invalid = cfg; invalid.blocks_per_node = -2;
    must_throw([&] { construct(invalid); }, "positive");
    invalid = cfg; invalid.execution_mode = static_cast<FWExecutionMode>(99);
    must_throw([&] { construct(invalid); }, "unknown FW execution mode");
    invalid = cfg; invalid.gpu_block_threads = 33;
    must_throw([&] { construct(invalid); }, cooperative ? "multiple of 32" : "cooperative launch");
    invalid = cfg; invalid.gpu_block_threads = 256;
    must_throw([&] { construct(invalid); }, cooperative ? "exceeds" : "cooperative launch");
    invalid = cfg; invalid.dicg_oracle_parallelism = DeviceResidentConfig::DICGOracleParallelism::Paired;
    must_throw([&] { construct(invalid); }, "cannot combine");
    invalid = cfg; invalid.blocks_per_node = std::numeric_limits<int>::max();
    must_throw([&] { construct(invalid); }, cooperative ? "complete node group" : "cooperative launch");
    must_throw([&] {
        DeviceResidentHybridSolver<LegacyObjective, FrankWolfeSecant> solver(
            problem, {}, {}, branch, cfg);
    }, "grad_multiblock");
    if (!cooperative) {
        must_throw([&] { construct(cfg); }, "cooperative launch");
        return;
    }
    must_throw([&] {
        DeviceResidentHybridSolver<OversizedWorkspaceObjective, FrankWolfeSecant> oversized(
            problem, {}, {}, branch, cfg);
    }, "workspace size overflows");
    DeviceResidentHybridSolver<SumObjective, FrankWolfeSecant> solver(
        problem, {}, {}, branch, cfg);
    const int capacity = solver.effective_gpu_batch_size();
    require(capacity > 0 && capacity < cfg.gpu_batch_size && solver.effective_num_streams() == 1,
            "cooperative batch/stream cap not applied");
    int submitted = 0, launches = 0, largest = 0;
    solver.set_scheduling_callback([&](const DRSSchedulingEvent& event) {
        if (event.type == DRSSchedulingEventType::Dispatched) {
            require(event.n_nodes <= capacity, "oversized cooperative batch");
            submitted += event.n_nodes; ++launches;
            largest = std::max(largest, event.n_nodes);
        }
    });
    // Infeasible nodes cannot prune the remaining frontier, ensuring all
    // capacity+1 nodes are submitted as one full and one partial batch.
    for (int j = 0; j <= capacity; ++j) {
        Node node(j, -1, 0, problem.lb, problem.ub);
        node.lb[0] = 2;
        solver.push(node);
    }
    solver.solve();
    require(submitted == capacity+1 && launches == 2 && largest == capacity,
            "cooperative capacity boundary/partial batch scheduling failed");
    std::cout << "K=" << k << " cooperative node capacity=" << capacity << '\n';
    cfg.gpu_batch_size = 2;
    DeviceResidentHybridSolver<SumObjective, FrankWolfeSecant> relaxation(
        problem, {}, {}, branch, cfg);
    relaxation.push(Node(0, -1, 0, problem.lb, problem.ub));
    relaxation.solve();
    require_near(relaxation.incumbent_value(), 0, 1e-8, "end-to-end relaxation failed");
}

// A linear objective reaches the CPU oracle optimum exactly, exercising
// dispatch, per-node LMO scratch allocation and the solver's partial batch.
template<int Dimension>
struct OracleLinearObjective : DeviceObjectiveNoPrepare {
    static constexpr int n_vars = Dimension, block_threads = 128, shared_bytes = 0;
    __device__ void grad(int tid, const double*, double* g, unsigned char*) const {
        for (int j = tid; j < n_vars; j += blockDim.x) g[j] = double(j % 7) - 3;
    }
    __device__ void grad_multiblock(MultiBlockContext& ctx, const double*,
        double* g, unsigned char*) const {
        for (int j = ctx.global_tid; j < n_vars; j += ctx.total_threads)
            g[j] = double(j % 7) - 3;
    }
    __device__ double value_local(int tid, const double* x, unsigned char*) const {
        double value = 0;
        for (int j = tid; j < n_vars; j += blockDim.x) value += (double(j % 7)-3)*x[j];
        return value;
    }
};

template<class LMO, int Dimension = 16>
void oracle_solver_integration(int k, const LMO& lmo) {
    using Objective = OracleLinearObjective<Dimension>;
    constexpr int n = Objective::n_vars;
    Problem problem{n, std::vector<double>(n, 0), std::vector<double>(n, 1), {}};
    std::vector<double> g(n), v(n);
    for (int j = 0; j < n; ++j) g[j] = double(j % 7)-3;
    lmo.compute_v_cpu(g, problem.lb, problem.ub, v);
    double expected = 0;
    for (int j = 0; j < n; ++j) expected += g[j]*v[j];
    BranchStrategy branch = [](const Node&, int&, const std::vector<double>&,
        const std::vector<int>&) { return std::vector<Node>{}; };
    DeviceResidentConfig cfg;
    cfg.execution_mode = FWExecutionMode::MultiBlock;
    cfg.blocks_per_node = k;
    cfg.gpu_batch_size = 4;
    cfg.gpu_max_iters = 10;
    cfg.gpu_eps = cfg.gpu_eps_min = 1e-8;
    DeviceResidentHybridSolver<Objective, FrankWolfeOpenLoop, LMO>
        solver(problem, {}, {}, branch, cfg, lmo);
    solver.push(Node(0, -1, 0, problem.lb, problem.ub));
    solver.solve();
    require_near(solver.incumbent_value(), expected, 1e-8, "multi-block LMO solver optimum mismatch");
}

int main() {
    try { launch_arithmetic(); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        int device, coop, sms;
        DRS_CUDA_CHECK(cudaGetDevice(&device));
        DRS_CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device));
        if (!coop) { solver_integration(2, false); return 77; }
        DRS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        for (int k : {1, 2, 4, 8}) {
            int resident;
            DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
                drs_fw_multiblock_kernel<SumObjective, FrankWolfeSecant, BoxLMO>, 128, 32));
            if (resident * sms < k) {
                std::cout << "SKIP K=" << k << ": no complete resident node group\n";
                continue;
            }
            solver_integration(k, true);
            oracle_solver_integration(k, UnitSimplexLMO<16>{});
            oracle_solver_integration(k, EqualitySimplexLMO<16>{});
            oracle_solver_integration(k, BirkhoffLMO<4>{});
            oracle_solver_integration<UnitSimplexLMO<257>, 257>(k, {});
            oracle_solver_integration<EqualitySimplexLMO<257>, 257>(k, {});
            oracle_solver_integration<BirkhoffLMO<129>, 129*129>(k, {});
            numerical<FrankWolfeOpenLoop, CooperativeBoxLMO>(k, 100, 1e-8, FrankWolfeOpenLoop{});
            numerical<FrankWolfeSecant, CooperativeBoxLMO>(k, 20, 1e-8, FrankWolfeSecant{});
            numerical<FrankWolfeOpenLoop, CooperativeBoxLMO>(k, 5, 1e-8, FrankWolfeOpenLoop{}, true);
            numerical(k, 100, 1e8, FrankWolfeOpenLoop{});
            numerical(k, 100, 1e-8, FrankWolfeOpenLoop{}, false, true);
            numerical(k, 5, 1e-8, FrankWolfeOpenLoop{}, false, false, true);
            numerical(k, 20, 1e-8, FrankWolfe<QuadraticExactLS>{});
            numerical(k, 20, 1e-8, DICGSecant{});
            numerical(k, 5, 1e-8, FrankWolfeOpenLoop{}, true);
            numerical(k, 20, 1e-8, DICGOpenLoop{});
            numerical(k, 1, 1e-12, FrankWolfeOpenLoop{});
            numerical(k, 100, 1e-8, FrankWolfeOpenLoop{});
            numerical(k, 20, 1e-8, FrankWolfeSecant{});
            int active;
            DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, stress, 128, 0));
            const int nodes = std::min(3, active * sms / k);
            if (!nodes) continue;
            NodeBarrier* barriers; unsigned* slots; int* errors; unsigned* progress;
            DRS_CUDA_CHECK(cudaMalloc(&barriers, nodes * sizeof(NodeBarrier)));
            DRS_CUDA_CHECK(cudaMalloc(&slots, nodes * k * 128 * sizeof(unsigned)));
            DRS_CUDA_CHECK(cudaMalloc(&errors, sizeof(int)));
            DRS_CUDA_CHECK(cudaMalloc(&progress, sizeof(unsigned)));
            DRS_CUDA_CHECK(cudaMemset(progress, 0, sizeof(unsigned)));
            DRS_CUDA_CHECK(cudaMemset(barriers, 0, nodes * sizeof(NodeBarrier)));
            DRS_CUDA_CHECK(cudaMemset(errors, 0, sizeof(int)));
            void* args[] = {&barriers, &slots, &errors, &k, &progress};
            DRS_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)stress, dim3(nodes*k), dim3(128), args));
            DRS_CUDA_CHECK(cudaDeviceSynchronize());
            int count;
            DRS_CUDA_CHECK(cudaMemcpy(&count, errors, sizeof(int), cudaMemcpyDeviceToHost));
            DRS_CUDA_CHECK(cudaFree(barriers));
            DRS_CUDA_CHECK(cudaFree(slots));
            DRS_CUDA_CHECK(cudaFree(errors));
            DRS_CUDA_CHECK(cudaFree(progress));
            if (count) throw std::runtime_error("barrier visibility failure");
            std::cout << "K=" << k << " barrier generations passed\n";
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

#else
int main() { return 77; }
#endif
