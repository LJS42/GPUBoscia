#include <gpuboscia/device_resident_solver.cuh>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if DRS_HAS_MULTIBLOCK_ATOMICS
namespace {
constexpr int cases = 3;
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t n) { DRS_CUDA_CHECK(cudaMalloc(&data, n * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
};
template<class LMO, bool Multi, bool LeaderOnly = false>
__global__ void oracle(LMO lmo, int n, int k, const double* g,
    const double* lb, const double* ub, double* v, DRSNodeControl* controls, int repeats) {
    const int node = Multi ? blockIdx.x / k : blockIdx.x;
    MultiBlockContext ctx(node, Multi ? blockIdx.x % k : 0, k, &controls[node].barrier);
    ctx.lmo_workspace = reinterpret_cast<unsigned char*>(controls)
        + drs_multiblock_scratch_offset(cases, k) + node * DRSLMOWorkspace<LMO>::bytes;
    const size_t offset = size_t(node) * n;
    for (int r = 0; r < repeats; ++r) {
        if constexpr (Multi) {
            if constexpr (LeaderOnly || !DRSHasMultiBlockLMO<LMO>::value) {
                if (ctx.block_rank == 0)
                    lmo.compute_v(threadIdx.x, n, g + offset, lb + offset, ub + offset, v + offset);
            } else {
                lmo.compute_v_multiblock(ctx, n, g + offset, lb + offset, ub + offset, v + offset);
            }
            ctx.node_sync();
        } else {
            lmo.compute_v(threadIdx.x, n, g + offset, lb + offset, ub + offset, v + offset);
            __syncthreads();
        }
    }
}

void check(const std::vector<double>& actual, const std::vector<double>& expected,
           const std::string& label) {
    for (size_t j = 0; j < actual.size(); ++j)
        if (!std::isfinite(actual[j]) || std::abs(actual[j] - expected[j]) > 1e-9)
            throw std::runtime_error(label + ": vertex mismatch at " + std::to_string(j));
}

template<class LMO>
void check_barriers(const LMO&, DRSNodeControl*, int, bool) {}

template<int N, bool Equality>
void check_barriers(const UnitSimplexLMO<N, Equality>&, DRSNodeControl* controls, int k, bool leader) {
    using LMO = UnitSimplexLMO<N, Equality>;
    unsigned expected = 0;
    if (k > 1) {
        expected = 1; // One caller barrier per invocation.
        if (!leader && LMO::sort_size > LMO::multiblock_tile_size) {
            expected += 2; // Publish local runs/mass and tile scans.
            for (int width = LMO::multiblock_tile_size; width < LMO::sort_size; width <<= 1)
                ++expected; // One per merge level.
        }
        expected *= 2; // launch(2) below.
    }
    std::vector<DRSNodeControl> state(cases);
    DRS_CUDA_CHECK(cudaMemcpy(state.data(), controls, cases*sizeof(DRSNodeControl), cudaMemcpyDeviceToHost));
    for (const auto& node : state)
        if (node.barrier.generation != expected)
            throw std::runtime_error("unexpected simplex node barrier count");
}

template<class LMO>
void run(const char* name, const LMO& lmo, int n,
    const std::vector<double>& g, const std::vector<double>& lb,
    const std::vector<double>& ub, bool timing, int threads = 128) {
    std::vector<double> expected(cases*n), baseline(cases*n), actual(cases*n);
    for (int c = 0; c < cases; ++c) {
        auto first = size_t(c)*n;
        std::vector<double> cg(g.begin()+first, g.begin()+first+n);
        std::vector<double> cl(lb.begin()+first, lb.begin()+first+n);
        std::vector<double> cu(ub.begin()+first, ub.begin()+first+n), cv(n);
        lmo.compute_v_cpu(cg, cl, cu, cv);
        std::copy(cv.begin(), cv.end(), expected.begin()+first);
    }
    Buffer<double> dg(g.size()), dl(lb.size()), du(ub.size()), dv(expected.size());
    DRS_CUDA_CHECK(cudaMemcpy(dg.data, g.data(), g.size()*sizeof(double), cudaMemcpyHostToDevice));
    DRS_CUDA_CHECK(cudaMemcpy(dl.data, lb.data(), lb.size()*sizeof(double), cudaMemcpyHostToDevice));
    DRS_CUDA_CHECK(cudaMemcpy(du.data, ub.data(), ub.size()*sizeof(double), cudaMemcpyHostToDevice));
    int device, sms, cooperative;
    DRS_CUDA_CHECK(cudaGetDevice(&device));
    DRS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    DRS_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
    for (int mode : {0, 1, 2}) for (int requested : {1, 2, 4, 8}) {
        if (mode == 0 && requested != 1) continue;
        if (mode == 1 && requested == 1) continue;
        const int k = mode == 0 ? 0 : requested;
        const bool leader = mode == 1;
        const char* mode_name = mode == 0 ? "single" : (leader ? "leader" : "multi");
        const int blocks = k ? k : 1;
        if (k) {
            if (!cooperative) continue;
            int active;
            if (leader) {
                DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &active, oracle<LMO, true, true>, threads, 0));
            } else {
                DRS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &active, oracle<LMO, true>, threads, 0));
            }
            if (active*sms < cases*k) {
                std::cerr << "SKIP " << name << " K=" << k << ": insufficient residency\n";
                continue;
            }
        }
        const size_t bytes = drs_multiblock_control_bytes(cases, blocks, DRSLMOWorkspace<LMO>::bytes);
        Buffer<unsigned char> storage(bytes);
        auto controls = reinterpret_cast<DRSNodeControl*>(storage.data);
        DRS_CUDA_CHECK(cudaMemset(storage.data, 0, bytes));
        const auto launch = [&](int repeats) {
            if (k) {
                // Local copies let the CUDA launch API take addresses of values.
                LMO arg_lmo = lmo;
                int arg_n = n, arg_k = k;
                void* args[] = {&arg_lmo, &arg_n, &arg_k, &dg.data, &dl.data, &du.data,
                    &dv.data, &controls, &repeats};
                if (leader) {
                    DRS_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)oracle<LMO, true, true>,
                        dim3(cases*k), dim3(threads), args));
                } else {
                    DRS_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)oracle<LMO, true>,
                        dim3(cases*k), dim3(threads), args));
                }
            } else {
                oracle<LMO, false><<<cases, threads>>>(lmo, n, 1, dg.data, dl.data, du.data,
                    dv.data, controls, repeats);
                DRS_CUDA_CHECK(cudaGetLastError());
            }
        };
        launch(2); // Also checks workspace reuse; every case has different input.
        DRS_CUDA_CHECK(cudaDeviceSynchronize());
        DRS_CUDA_CHECK(cudaMemcpy(actual.data(), dv.data, actual.size()*sizeof(double), cudaMemcpyDeviceToHost));
        check(actual, expected, std::string(name)+" CPU K="+std::to_string(k));
        check_barriers(lmo, controls, k, leader);
        if (!k) baseline = actual;
        else check(actual, baseline, std::string(name)+" single K="+std::to_string(k));
        if (timing) {
            cudaEvent_t start, end;
            DRS_CUDA_CHECK(cudaEventCreate(&start));
            DRS_CUDA_CHECK(cudaEventCreate(&end));
            const int repeats = n > 16384 ? 1 : 10;
            constexpr int samples = 5;
            std::vector<float> times;
            for (int sample = 0; sample < samples; ++sample) {
                DRS_CUDA_CHECK(cudaEventRecord(start));
                launch(repeats);
                DRS_CUDA_CHECK(cudaEventRecord(end));
                DRS_CUDA_CHECK(cudaEventSynchronize(end));
                float ms;
                DRS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
                times.push_back(ms*1000/repeats);
            }
            std::sort(times.begin(), times.end());
            int owners = 1;
            if constexpr (DRSHasMultiBlockLMO<LMO>::value)
                if (!leader && k) owners = lmo.multiblock_effective_blocks(blocks, threads);
            std::cout << name << ',' << n << ',' << mode_name << ','
                << blocks << ',' << owners << ',' << threads << ',' << cases << ',' << DRSLMOWorkspace<LMO>::bytes << ','
                << times[samples/2] << '\n';
            DRS_CUDA_CHECK(cudaEventDestroy(start));
            DRS_CUDA_CHECK(cudaEventDestroy(end));
        }
    }
}

template<int N, bool Equality>
void simplex(bool timing, int threads = 128) {
    UnitSimplexLMO<N, Equality> lmo;
    lmo.right_side = 7.0;
    for (int j = 0; j < N; ++j) lmo.is_int[j] = j % 3 != 0;
    std::mt19937 rng(42);
    std::vector<double> g(cases*N), lb(cases*N), ub(cases*N, 3.0);
    for (int c = 0; c < cases; ++c) for (int j = 0; j < N; ++j) {
        const int e = c*N+j;
        // Ties, all-positive costs, mixed integer/continuous, forced lower
        // bounds, and zero capacities. All arithmetic uses exact integers.
        g[e] = c == 1 ? 1.0 + rng()%4 : int(rng()%9)-4;
        if (c == 2 && j < 3 && lmo.is_int[j]) lb[e] = 1;
        if (j % 5 == 0 && lmo.is_int[j]) ub[e] = lb[e];
    }
    run(Equality ? "equality_simplex" : "simplex", lmo, N, g, lb, ub, timing, threads);
    // Fully integer allocation with saturated mass and deterministic ties.
    for (int j = 0; j < N; ++j) lmo.is_int[j] = 1;
    std::fill(lb.begin(), lb.end(), 0.0);
    std::fill(ub.begin(), ub.end(), 1.0);
    lmo.right_side = N;
    run(Equality ? "equality_integer" : "simplex_integer", lmo, N, g, lb, ub, timing, threads);
}

template<int Order>
void birkhoff(bool timing, int threads = 128) {
    constexpr int n = Order*Order;
    std::mt19937 rng(123);
    std::vector<double> g(cases*n), lb(cases*n), ub(cases*n, 1.0);
    for (int c = 0; c < cases; ++c) for (int j = 0; j < n; ++j)
        g[c*n+j] = c == 1 ? 0.0 : int(rng()%17)-8;
    // Preserve an identity matching while excluding some edges.
    for (int r = 0; r < Order; ++r) for (int c = 0; c < Order; ++c)
        if (r != c && (r+c)%3 == 0) ub[2*n+r*Order+c] = 0;
    lb[2*n] = 1; // Forced assignment, residual Hungarian problem.
    run("birkhoff", BirkhoffLMO<Order>{}, n, g, lb, ub, timing, threads);
    // No complete matching, without conflicting forced assignments.
    for (int c = 0; c < cases; ++c)
        for (int j = 0; j < Order; ++j) ub[c*n+j] = 0;
    std::fill(lb.begin(), lb.end(), 0.0);
    run("birkhoff_infeasible", BirkhoffLMO<Order>{}, n, g, lb, ub, false, threads);
}
} // namespace
#endif

int main(int argc, char** argv) {
#if DRS_HAS_MULTIBLOCK_ATOMICS
    int devices;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        int device, cooperative;
        DRS_CUDA_CHECK(cudaGetDevice(&device));
        DRS_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
        if (!cooperative) {
            std::cerr << "SKIP: multi-block LMO requires cooperative launch support\n";
            return 77;
        }
        const bool timing = !(argc > 1 && std::string(argv[1]) == "--test");
        if (timing) std::cout << "lmo,n,mode,K,lmo_blocks,threads,nodes,reserved_scratch_bytes_per_node,batch_us\n";
        simplex<17, false>(timing);
        simplex<17, true>(timing);
        simplex<257, false>(timing);
        simplex<257, true>(timing);
        birkhoff<5>(timing);
        birkhoff<32>(timing);
        birkhoff<129>(timing); // Larger block-local Hungarian workload.
        if (!timing) {
            simplex<257, false>(false, 32);
            simplex<257, true>(false, 96);
            birkhoff<33>(false, 32);
            birkhoff<97>(false, 96);
        }
        if (argc > 1 && std::string(argv[1]) == "--large") {
            simplex<1025, false>(timing);
            simplex<1025, true>(timing);
            birkhoff<256>(timing);
        }
        if (!timing) std::cout << "LMO CPU/single/multi comparisons passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
#else
    return 77;
#endif
}
