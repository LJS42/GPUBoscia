// GPUBoscia graph-isomorphism example over the Birkhoff polytope.
//
//   minimize 0.5 ||B X - X A||_F^2
//   subject to X is a permutation matrix.
//
// A is loaded from the configured MAT instance. B is obtained by a random
// permutation, so the generated pair is isomorphic and has optimum zero.
// Pass --seed N to reproduce a generated pair exactly.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>
#include <cuda_runtime.h>

#include <gpuboscia/branching.h>
#include <gpuboscia/gpuboscia.cuh>
#include <gpuboscia/line_search_options.h>
#include "common/options.h"
#include <gpuboscia/logger.h>

#define MAIN_CUDA_CHECK(call) do {                                             \
    const cudaError_t error_ = (call);                                         \
    if (error_ != cudaSuccess) {                                               \
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error_)); \
        return 1;                                                              \
    }                                                                          \
} while (false)

struct MatElement { std::uint32_t type = 0, bytes = 0; const unsigned char* data = nullptr; };

static std::uint32_t read_u32(const unsigned char* p) {
    std::uint32_t value; std::memcpy(&value, p, sizeof(value)); return value;
}

static MatElement next_element(const unsigned char*& p, const unsigned char* end) {
    if (end - p < 4) throw std::runtime_error("truncated MAT element");
    const std::uint32_t tag = read_u32(p);
    MatElement e;
    if (tag >> 16) {
        e.type = tag & 0xffffu; e.bytes = tag >> 16; e.data = p + 4; p += 8;
    } else {
        if (end - p < 8) throw std::runtime_error("truncated MAT tag");
        e.type = tag; e.bytes = read_u32(p + 4); e.data = p + 8;
        // MATLAB writers commonly omit 8-byte padding after miCOMPRESSED.
        const std::size_t stored = e.type == 15
            ? e.bytes : (e.bytes + 7u) & ~std::size_t(7u);
        if ((std::size_t)(end - e.data) < stored)
            throw std::runtime_error("truncated MAT payload");
        p = e.data + stored;
    }
    return e;
}

static std::vector<double> parse_mat_matrix(
    const unsigned char* data, std::size_t size, const std::string& wanted,
    int& rows_out, int& cols_out)
{
    const unsigned char* p = data;
    const unsigned char* end = data + size;
    MatElement matrix = next_element(p, end);
    if (matrix.type != 14) throw std::runtime_error("expected miMATRIX");
    p = matrix.data; end = matrix.data + matrix.bytes;
    (void)next_element(p, end); // array flags
    MatElement dims = next_element(p, end);
    if (dims.bytes < 8) throw std::runtime_error("invalid MAT dimensions");
    const int rows = (int)read_u32(dims.data);
    const int cols = (int)read_u32(dims.data + 4);
    MatElement name = next_element(p, end);
    const std::string variable((const char*)name.data, name.bytes);
    MatElement real = next_element(p, end);
    if (variable != wanted) return {};
    std::vector<double> result((std::size_t)rows * cols);
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r) {
            const std::size_t source = (std::size_t)c * rows + r;
            double value = 0.0;
            if (real.type == 2) value = real.data[source];             // miUINT8
            else if (real.type == 9) std::memcpy(&value, real.data + 8 * source, 8);
            else throw std::runtime_error("unsupported MAT numeric type");
            result[(std::size_t)r * cols + c] = value;
        }
    rows_out = rows; cols_out = cols;
    return result;
}

static std::vector<double> load_mat_A(const std::string& path, int& rows, int& cols) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open instance: " + path);
    std::vector<unsigned char> file(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (file.size() < 136 || file[126] != 'I' || file[127] != 'M')
        throw std::runtime_error("unsupported MAT v5 byte order");
    const unsigned char* p = file.data() + 128;
    const unsigned char* end = file.data() + file.size();
    while (p < end) {
        MatElement outer = next_element(p, end);
        std::vector<unsigned char> inflated;
        const unsigned char* payload = outer.data;
        std::size_t payload_size = outer.bytes;
        if (outer.type == 15) {
            uLongf capacity = std::max<uLongf>(outer.bytes * 8u, 1024u);
            int status;
            do {
                inflated.resize(capacity);
                uLongf actual = capacity;
                status = uncompress(inflated.data(), &actual, outer.data, outer.bytes);
                if (status == Z_BUF_ERROR) capacity *= 2;
                else { inflated.resize(actual); break; }
            } while (capacity < (1u << 28));
            if (status != Z_OK) throw std::runtime_error("cannot inflate MAT data");
            payload = inflated.data(); payload_size = inflated.size();
        } else if (outer.type != 14) continue;
        auto values = parse_mat_matrix(payload, payload_size, "A", rows, cols);
        if (!values.empty()) return values;
    }
    throw std::runtime_error("MAT file has no matrix A");
}

template<int Order, int Threads>
struct GraphIsoObjective {
    static constexpr bool supports_quadratic_exact = true;
    static constexpr int n_vars = Order * Order;
    static constexpr int block_threads = Threads;
    static constexpr int shared_bytes = n_vars * (int)sizeof(double) + 16;
    const double* d_A; const double* d_B;
    const double* h_A; const double* h_B;

    __host__ __device__ double residual_entry(
        const double* A, const double* B, const double* x, int r, int c) const {
        double value = 0.0;
        for (int k = 0; k < Order; ++k)
            value += B[r*Order+k] * x[k*Order+c]
                   - x[r*Order+k] * A[k*Order+c];
        return value;
    }
    double f(const std::vector<double>& x) const {
        double value = 0.0;
        for (int r = 0; r < Order; ++r) for (int c = 0; c < Order; ++c) {
            const double z = residual_entry(h_A, h_B, x.data(), r, c);
            value += 0.5 * z * z;
        }
        return value;
    }
    void gradient_host(const std::vector<double>& x, std::vector<double>& g) const {
        double residual[n_vars];
        for (int r=0;r<Order;++r) for(int c=0;c<Order;++c)
            residual[r*Order+c]=residual_entry(h_A,h_B,x.data(),r,c);
        for (int p=0;p<Order;++p) for(int q=0;q<Order;++q) {
            double value=0.0;
            for(int r=0;r<Order;++r) value += h_B[r*Order+p]*residual[r*Order+q];
            for(int c=0;c<Order;++c) value -= residual[p*Order+c]*h_A[q*Order+c];
            g[p*Order+q]=value;
        }
    }
    __device__ double* residual_buffer(unsigned char* smem) const {
        return reinterpret_cast<double*>((reinterpret_cast<uintptr_t>(smem)+15u)&~uintptr_t(15u));
    }
    __device__ void prepare_block(int, unsigned char*) const {}
    __device__ void grad(int tid, const double* x, double* g, unsigned char* smem) const {
        double* residual=residual_buffer(smem);
        for(int e=tid;e<n_vars;e+=blockDim.x)
            residual[e]=residual_entry(d_A,d_B,x,e/Order,e%Order);
        __syncthreads();
        for(int e=tid;e<n_vars;e+=blockDim.x) {
            const int p=e/Order,q=e%Order; double value=0.0;
            for(int r=0;r<Order;++r) value += d_B[r*Order+p]*residual[r*Order+q];
            for(int c=0;c<Order;++c) value -= residual[p*Order+c]*d_A[q*Order+c];
            g[e]=value;
        }
        __syncthreads();
    }
    __device__ double quadratic_curvature_local(
        int tid, const double* dir, unsigned char*) const {
        double curvature = 0.0;
        for (int e = tid; e < n_vars; e += blockDim.x) {
            const double r = residual_entry(d_A, d_B, dir, e/Order, e%Order);
            curvature += r*r;
        }
        return curvature;
    }
    __device__ double value_local(int tid,const double* x,unsigned char*) const {
        double value=0.0;
        for(int e=tid;e<n_vars;e+=blockDim.x) {
            const double z=residual_entry(d_A,d_B,x,e/Order,e%Order);
            value += 0.5*z*z;
        }
        return value;
    }
};

int main(int argc, char** argv) {
    GPUBosciaExampleOptions opts;
    const int cli_status=parse_gpuboscia_example_options(argc,argv,opts);
    if(cli_status>=0) return cli_status;
    constexpr int order=57, variables=order*order, threads=832;
    const std::string path=opts.instance_path.empty()
        ? std::string(GPUBOSCIA_SOURCE_DIR)+"/test_instances/sts_19_57.mat"
        : opts.instance_path;
    int rows=0,cols=0; std::vector<double> A;
    try { A=load_mat_A(path,rows,cols); }
    catch(const std::exception& e){std::fprintf(stderr,"Instance error: %s\n",e.what());return 1;}
    if(rows!=order||cols!=order){std::fprintf(stderr,"This binary requires a %dx%d instance; got %dx%d.\n",order,order,rows,cols);return 1;}

    const std::uint64_t random_seed = opts.random_seed_explicit
        ? opts.random_seed
        : (static_cast<std::uint64_t>(std::random_device{}()) << 32)
          ^ static_cast<std::uint64_t>(std::random_device{}())
          ^ static_cast<std::uint64_t>(
              std::chrono::high_resolution_clock::now()
                  .time_since_epoch().count());
    std::mt19937_64 rng(random_seed);
    std::vector<int> permutation(order); std::iota(permutation.begin(),permutation.end(),0);
    std::shuffle(permutation.begin(),permutation.end(),rng);
    std::vector<double> B(order*order);
    for(int r=0;r<order;++r) for(int c=0;c<order;++c)
        B[r*order+c]=A[permutation[r]*order+permutation[c]];

    double *d_A=nullptr,*d_B=nullptr;
    MAIN_CUDA_CHECK(cudaMalloc(&d_A,A.size()*sizeof(double)));
    MAIN_CUDA_CHECK(cudaMalloc(&d_B,B.size()*sizeof(double)));
    MAIN_CUDA_CHECK(cudaMemcpy(d_A,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice));
    MAIN_CUDA_CHECK(cudaMemcpy(d_B,B.data(),B.size()*sizeof(double),cudaMemcpyHostToDevice));
    using Objective=GraphIsoObjective<order,threads>;
    Objective objective{d_A,d_B,A.data(),B.data()};
    Problem problem; problem.n_vars=variables;
    problem.lb.assign(variables,0.0); problem.ub.assign(variables,1.0);
    problem.int_vars.resize(variables); std::iota(problem.int_vars.begin(),problem.int_vars.end(),0);
    problem.f=[objective](const std::vector<double>& x){return objective.f(x);};
    problem.grad_f=[objective](const std::vector<double>& x,std::vector<double>& g){objective.gradient_host(x,g);};

    // A 100x100 instance has 10,000 variables. Keep only a small number of
    // nodes resident so the per-node global workspace does not exhaust VRAM.
    DeviceResidentConfig cfg; cfg.gpu_batch_size=128; cfg.num_streams=2;
    cfg.gpu_max_iters=500; cfg.gpu_eps=opts.gpu_eps; cfg.gpu_eps_decay=opts.gpu_eps_decay;
    cfg.gpu_eps_min=opts.gpu_eps_min; cfg.serial_iters=1000; cfg.integer_tol=1e-5;
    cfg.abs_gap_tol=opts.abs_gap_tol; cfg.rel_gap_tol=opts.rel_gap_tol;
    BirkhoffLMO<order> lmo;
    Logger logger(true,opts.log_interval_nodes);
    auto settings=gpuboscia::create_default_settings(); settings.device=cfg;
    settings.algorithm=opts.algorithm; settings.line_search=opts.line_search;
    settings.time_limit_seconds=opts.time_limit_s;
    // The graph-isomorphism objective is a squared Frobenius norm, so zero
    // is a valid global lower bound. A CLI value may override this default.
    settings.natural_lower_bound=0.0;
    settings.progress=[&logger](int i,int o,double b,double incumbent,bool fresh){logger.print_row(i,o,b,incumbent,fresh);};
    LoggerParams lp; lp.traversal_strategy="best_first (shared queue)";
    lp.branching_strategy="branch_pow2(k=3, binary)"; lp.algorithm=fw_algorithm_name(settings.algorithm);
    lp.line_search_method=line_search_name(settings.line_search.kind); lp.n_vars=variables;
    lp.n_int_vars=variables; lp.abs_gap_tol=cfg.abs_gap_tol; lp.rel_gap_tol=cfg.rel_gap_tol;
    lp.time_limit_s=settings.time_limit_seconds;
    lp.natural_lower_bound=settings.natural_lower_bound;
    lp.random_seed=random_seed; lp.print_random_seed=true;
    logger.print_header(lp);

    const gpuboscia::Result result=gpuboscia::solve(problem,objective,lmo,settings);
    SolutionStats stats; stats.primal_objective=result.objective; stats.dual_bound=result.dual_bound;
    stats.total_nodes=result.nodes_processed; stats.total_time_s=result.elapsed_seconds;
    stats.solution=result.x;
    stats.solution_status=result.reached_time_limit?"Time limit reached":(result.reached_tolerance?"Reached tolerance":"Optimal");
    logger.print_footer(stats);
    bool isomorphic=result.x.size()==variables && result.objective<=1e-6;
    std::printf("\nGraph isomorphism example:\n  Instance:             %s\n  Random seed:          %llu\n  Vertices:             %d\n  Objective:            %.10g\n  Isomorphism found:    %s\n",
                path.c_str(), static_cast<unsigned long long>(random_seed),
                order,result.objective,isomorphic?"yes":"no");
    cudaFree(d_A); cudaFree(d_B); return 0;
}
