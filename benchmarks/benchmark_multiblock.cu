// Synthetic nonseparable workload. Timings describe this workload only.
#include <gpuboscia/device_resident_solver.cuh>
#include <iostream>
#include <iomanip>
#include <memory>

#if DRS_HAS_MULTIBLOCK_ATOMICS

template<int N>
struct BenchmarkObjective {
    static constexpr int n_vars = N, block_threads = 256, shared_bytes = 0;
    static constexpr bool supports_quadratic_exact = true;
    size_t multiblock_workspace_bytes(int k, int) const { return k * sizeof(double); }
    __device__ void prepare_block(int, unsigned char*) const {}
    __device__ void grad(int tid, const double* x, double* g, unsigned char*) const {
        double sum = 0;
        for (int j = tid; j < N; j += blockDim.x) sum += x[j] - .3;
        sum = feasibility_block_sum(sum);
        for (int j = tid; j < N; j += blockDim.x) g[j] = x[j] - .3 + sum/N;
    }
    __device__ void grad_multiblock(MultiBlockContext& ctx, const double* x,
                                    double* g, unsigned char* workspace) const {
        double sum = 0;
        for (int j = ctx.global_tid; j < N; j += ctx.total_threads) sum += x[j] - .3;
        sum = feasibility_block_sum(sum);
        auto partial = reinterpret_cast<double*>(workspace);
        if (ctx.thread_rank == 0) partial[ctx.block_rank] = sum;
        ctx.node_sync();
        sum = 0;
        for (int b = 0; b < ctx.block_count; ++b) sum += partial[b];
        for (int j = ctx.global_tid; j < N; j += ctx.total_threads) g[j] = x[j] - .3 + sum/N;
        ctx.node_sync();
    }
    __device__ double value_local(int tid, const double* x, unsigned char*) const {
        double sum = 0, value = 0;
        for (int j = tid; j < N; j += blockDim.x) sum += x[j] - .3;
        sum = feasibility_block_sum(sum);
        for (int j = tid; j < N; j += blockDim.x) {
            const double d = x[j] - .3;
            value += .5*d*(d + sum/N);
        }
        return value;
    }
    __device__ double quadratic_curvature_local(int tid, const double* d, unsigned char*) const {
        double sum = 0, value = 0;
        for (int j = tid; j < N; j += blockDim.x) sum += d[j];
        sum = feasibility_block_sum(sum);
        for (int j = tid; j < N; j += blockDim.x) value += d[j]*(d[j] + sum/N);
        return value;
    }
};

template<class T>
struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) { DRS_CUDA_CHECK(cudaMalloc(&data, count*sizeof(T))); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

template<bool Multi>
__device__ void phase_sync(const MultiBlockContext& ctx) {
    if constexpr (Multi) ctx.node_sync();
    else ctx.block_sync();
}

// Stage timings include the dependent phase barriers. They are standalone
// microbenchmarks; they are not attributed timings from the full B&B solve.
template<int N, bool Multi>
__global__ void stages(int k, NodeBarrier* barrier, double* x, double* g,
    double* v, const double* lb, const double* ub, unsigned char* workspace,
    unsigned long long* ticks) {
    MultiBlockContext ctx(0, blockIdx.x, k, barrier);
    BenchmarkObjective<N> obj;
    BoxLMO lmo;
    constexpr int repeats = 30;
    phase_sync<Multi>(ctx);
    auto start = clock64();
    for (int i = 0; i < repeats; ++i) {
        if constexpr (Multi) obj.grad_multiblock(ctx, x, g, workspace);
        else obj.grad(threadIdx.x, x, g, workspace);
        phase_sync<Multi>(ctx);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = (clock64()-start)/repeats;
    phase_sync<Multi>(ctx);
    start = clock64();
    for (int i = 0; i < repeats; ++i) {
        if constexpr (Multi) lmo.compute_v_multiblock(ctx, N, g, lb, ub, v);
        else lmo.compute_v(threadIdx.x, N, g, lb, ub, v);
        phase_sync<Multi>(ctx);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) ticks[1] = (clock64()-start)/repeats;
    phase_sync<Multi>(ctx);
    start = clock64();
    for (int i = 0; i < 20000; ++i) phase_sync<Multi>(ctx);
    if (blockIdx.x == 0 && threadIdx.x == 0) ticks[2] = (clock64()-start)/20000;
}

using BenchmarkAlgorithm = FrankWolfe<QuadraticExactLS>;

template<int N>
void measure(int k) {
    using Objective = BenchmarkObjective<N>;
    using Solver = DeviceResidentHybridSolver<Objective, BenchmarkAlgorithm>;
    const bool multi = k > 0;
    const int blocks = multi ? k : 1;
    int device, active, sms, clock_khz;
    DRS_CUDA_CHECK(cudaGetDevice(&device));
    DRS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    DRS_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
    if (multi) {
        DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
            drs_fw_multiblock_kernel<Objective, BenchmarkAlgorithm, BoxLMO>, 256, 64));
        if (active * sms < k) { std::cerr << "SKIP n=" << N << " K=" << k << " (capacity)\n"; return; }
    } else {
        DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
            drs_fw_relax_batch_kernel<Objective, BenchmarkAlgorithm, BoxLMO>, 256, 64));
    }
    DeviceResidentConfig cfg;
    cfg.execution_mode = multi ? FWExecutionMode::MultiBlock : FWExecutionMode::SingleBlock;
    cfg.blocks_per_node = blocks;
    cfg.gpu_batch_size = 4;
    cfg.num_streams = 1; // Same streams and requested batch for comparisons.
    cfg.gpu_max_iters = 200;
    cfg.gpu_eps = cfg.gpu_eps_min = 1e-5;
    Problem problem{N, std::vector<double>(N, 0), std::vector<double>(N, 1), {}};
    BranchStrategy branch = [](const Node& node, int& id, const std::vector<double>& x,
                               const std::vector<int>& iv) { return branch_pow2(node, id, x, iv, 1); };
    double latency_ms = 0, bnb_ms = 0;
    int completed = 0, batch = 0;
    size_t allocated = 0;
    bool limited = false;
    double objective = 0, dual_bound = 0, fw_gap = 0;
    // Setup and root creation are outside the solve timer. Each solve includes
    // transfers, scheduling, GPU FW, certificate processing, and pruning.
    for (int run = 0; run < 4; ++run) {
        problem.int_vars = run == 3 ? std::vector<int>{0,1,2,3} : std::vector<int>{};
        size_t before, total, after;
        DRS_CUDA_CHECK(cudaMemGetInfo(&before, &total));
        auto solver = std::make_unique<Solver>(problem, Objective{}, BenchmarkAlgorithm{}, branch,
                                               cfg, BoxLMO{}, 30.0);
        DRS_CUDA_CHECK(cudaMemGetInfo(&after, &total));
        allocated = before > after ? before-after : 0;
        batch = solver->effective_gpu_batch_size();
        solver->push(Node(0, -1, 0, problem.lb, problem.ub));
        const auto start = std::chrono::steady_clock::now();
        solver->solve();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-start).count();
        if (run == 1 || run == 2) latency_ms += ms/2; // First solve warms up.
        if (run == 3) {
            bnb_ms = ms; completed = solver->nodes_processed(); limited = solver->reached_time_limit();
            objective = solver->incumbent_value(); dual_bound = solver->dual_bound();
            fw_gap = solver->last_gpu_gap();
        }
    }
    // Five vectors for isolated stage timing, allocated after solver teardown.
    Buffer<double> vectors(size_t(5)*N);
    Buffer<unsigned char> workspace(std::max(size_t(16), blocks*sizeof(double)));
    Buffer<NodeBarrier> barrier(1);
    Buffer<unsigned long long> ticks(3);
    DRS_CUDA_CHECK(cudaMemset(vectors.data, 0, size_t(5)*N*sizeof(double)));
    DRS_CUDA_CHECK(cudaMemset(barrier.data, 0, sizeof(NodeBarrier)));
    auto x = vectors.data, g = x+N, v = g+N, lb = v+N, ub = lb+N;
    if (multi) {
        int stage_active;
        DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&stage_active, stages<N, true>, 256, 0));
        if (stage_active*sms < k) throw std::runtime_error("stage benchmark cooperative capacity exceeded");
        void* args[] = {&k, &barrier.data, &x, &g, &v, &lb, &ub, &workspace.data, &ticks.data};
        DRS_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)stages<N, true>, dim3(k), dim3(256), args));
    } else {
        stages<N, false><<<1, 256>>>(1, barrier.data, x, g, v, lb, ub, workspace.data, ticks.data);
        DRS_CUDA_CHECK(cudaGetLastError());
    }
    DRS_CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long measured[3];
    DRS_CUDA_CHECK(cudaMemcpy(measured, ticks.data, sizeof(measured), cudaMemcpyDeviceToHost));
    const double tick_us = 1000.0/clock_khz;
    std::cout << N << ',' << (multi ? "multi" : "single") << ',' << blocks << ',' << active
        << ',' << (multi ? active*sms/blocks : 0) << ',' << batch << ',' << allocated
        << ',' << latency_ms << ',' << bnb_ms << ',' << completed << ',' << completed*1000.0/bnb_ms
        << ',' << measured[0]*tick_us << ',' << measured[1]*tick_us << ',' << measured[2]*tick_us
        << ',' << limited << ',' << objective << ',' << dual_bound << ',' << fw_gap << '\n';
}

template<int N>
void dimension(bool cooperative) {
    measure<N>(0);
    if (cooperative) {
        for (int k : {1,2,4,8}) measure<N>(k);
        if constexpr (N >= 50000) measure<N>(16);
    }
}

int main(int argc, char** argv) {
    try {
        int device, cooperative;
        DRS_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        DRS_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        DRS_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
        std::cout << "# GPU=" << properties.name << "; clock64 estimates use nominal clockRate; stage timings include barriers\n";
        std::cout << "n,mode,K,active_blocks_per_sm,cooperative_node_limit,effective_batch,device_bytes,relaxation_ms,bnb_ms,nodes,nodes_per_second,gradient_us,lmo_us,barrier_us,time_limited,objective,dual_bound,last_fw_gap\n";
        std::cout << std::setprecision(9);
        dimension<5000>(cooperative);
        dimension<10000>(cooperative);
        dimension<50000>(cooperative);
        dimension<100000>(cooperative);
        dimension<1000000>(cooperative);
        if (argc > 1 && std::string(argv[1]) == "--large") dimension<10000000>(cooperative);
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
#else
int main() { std::cerr << "MultiBlock benchmark requires CUDA Toolkit 11.6+\n"; return 77; }
#endif
