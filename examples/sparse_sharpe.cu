// Discrete sparse Sharpe-ratio portfolio optimization with GPUBoscia.
//
//   minimize  0.5 v' (Sigma + epsilon I) v - mu' v
//   subject to 0<=v_i<=U*y_i, sum(y)<=m, y_i in {0,1}.
//
// The quadratic objective is the one used by the official mSSRM-PGA code;
// here GPUBoscia solves the exact bounded continuous-weight cardinality model.
// The portfolio is w=v/sum(v), computed only after optimization.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>
#include <cuda_runtime.h>

#include <gpuboscia/gpuboscia.cuh>
#include <gpuboscia/lmo.cuh>
#include <gpuboscia/logger.h>

#define CUDA_CHECK(call) do {                                                  \
    const cudaError_t e_ = (call);                                             \
    if (e_ != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e_));    \
        return 1;                                                              \
    }                                                                          \
} while (false)

namespace {
constexpr int MaxAssets = 125;
constexpr int Variables = 2 * MaxAssets;
// Kernels use storage for 400 variables.  The 224-thread block is one warp
// above the 200 active assets; block-stride loops cover all 256 sorting slots.
constexpr int Threads = 256;

struct MatElement {
    std::uint32_t type = 0, bytes = 0;
    const unsigned char* data = nullptr;
};

std::uint32_t u32(const unsigned char* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

MatElement next(const unsigned char*& p, const unsigned char* end) {
    if (end - p < 4) throw std::runtime_error("truncated MAT element");
    const std::uint32_t tag = u32(p);
    MatElement e;
    if (tag >> 16) {
        e.type = tag & 0xffffu;
        e.bytes = tag >> 16;
        e.data = p + 4;
        p += 8;
    } else {
        if (end - p < 8) throw std::runtime_error("truncated MAT tag");
        e.type = tag;
        e.bytes = u32(p + 4);
        e.data = p + 8;
        const std::size_t stored = e.type == 15
            ? e.bytes : (e.bytes + 7u) & ~std::size_t(7u);
        if ((std::size_t)(end - e.data) < stored)
            throw std::runtime_error("truncated MAT payload");
        p = e.data + stored;
    }
    return e;
}

std::vector<double> parse_data(const unsigned char* data, std::size_t size,
                               int& rows, int& cols) {
    const unsigned char* p = data;
    const unsigned char* end = data + size;
    // An miCOMPRESSED payload contains the complete miMATRIX element, whereas
    // an uncompressed top-level miMATRIX has already had its outer tag removed
    // by load_mat().  Accept both representations.
    if (size >= 8 && u32(data) == 14) {
        MatElement matrix = next(p, end);
        p = matrix.data;
        end = matrix.data + matrix.bytes;
    }
    (void)next(p, end);
    MatElement dims = next(p, end);
    rows = (int)u32(dims.data);
    cols = (int)u32(dims.data + 4);
    MatElement name = next(p, end);
    MatElement real = next(p, end);
    if (std::string((const char*)name.data, name.bytes) != "data") return {};
    if (real.type != 9) throw std::runtime_error("matrix 'data' must be double");
    std::vector<double> result((std::size_t)rows * cols);
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r)
            std::memcpy(&result[(std::size_t)r * cols + c],
                        real.data + 8 * ((std::size_t)c * rows + r), 8);
    return result;
}

std::vector<double> load_mat(const std::string& path, int& rows, int& cols) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open dataset: " + path);
    std::vector<unsigned char> file((std::istreambuf_iterator<char>(in)), {});
    if (file.size() < 136 || file[126] != 'I' || file[127] != 'M')
        throw std::runtime_error("only little-endian MAT v5 is supported");
    const unsigned char* p = file.data() + 128;
    const unsigned char* end = file.data() + file.size();
    while (p < end) {
        MatElement outer = next(p, end);
        std::vector<unsigned char> inflated;
        const unsigned char* payload = outer.data;
        std::size_t payload_size = outer.bytes;
        if (outer.type == 15) {
            uLongf cap = std::max<uLongf>(outer.bytes * 8u, 1024u);
            int status = Z_BUF_ERROR;
            while (status == Z_BUF_ERROR && cap < (1u << 29)) {
                inflated.resize(cap);
                uLongf actual = cap;
                status = uncompress(inflated.data(), &actual,
                                    outer.data, outer.bytes);
                if (status == Z_BUF_ERROR) cap *= 2;
                else inflated.resize(actual);
            }
            if (status != Z_OK) throw std::runtime_error("cannot inflate MAT data");
            payload = inflated.data();
            payload_size = inflated.size();
        } else if (outer.type != 14) {
            continue;
        }
        auto result = parse_data(payload, payload_size, rows, cols);
        if (!result.empty()) return result;
    }
    throw std::runtime_error("MAT file has no matrix named 'data'");
}

struct Options {
    std::string data = std::string(GPUBOSCIA_SOURCE_DIR)
        + "/DataSets/US_all_monthly.mat";
    int cardinality = 10;
    int observations = 60;
    int assets = 125;
    double epsilon = 1e-3;
    double time_limit = 3600.0;
    int threads = 256;
};

// Dense Cholesky factorization/solve for the host-side bound
// calculation.  H is positive definite because epsilon I is included.
std::vector<double> cholesky_factor(const std::vector<double>& H, int n) {
    std::vector<double> L((std::size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double value = H[(std::size_t)i * n + j];
            for (int k = 0; k < j; ++k)
                value -= L[(std::size_t)i*n+k] * L[(std::size_t)j*n+k];
            if (i == j) {
                if (!(value > 0.0))
                    throw std::runtime_error("covariance matrix is not positive definite");
                L[(std::size_t)i*n+j] = std::sqrt(value);
            } else {
                L[(std::size_t)i*n+j] = value / L[(std::size_t)j*n+j];
            }
        }
    return L;
}

std::vector<double> cholesky_solve(
    const std::vector<double>& L, const std::vector<double>& b, int n) {
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        double value = b[i];
        for (int j = 0; j < i; ++j) value -= L[(std::size_t)i*n+j] * y[j];
        y[i] = value / L[(std::size_t)i*n+i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double value = y[i];
        for (int j = i + 1; j < n; ++j) value -= L[(std::size_t)j*n+i] * x[j];
        x[i] = value / L[(std::size_t)i*n+i];
    }
    return x;
}

void usage(const char* program) {
    std::printf("Usage: %s [options]\n"
        "  --data PATH          MAT file containing 'data'\n"
        "  --cardinality M      optimize at most M of the candidates (default 10)\n"
        "  --observations T     use last T rows; 0 means all (default 60)\n"
        "  --assets N           use the first N asset columns (default 200)\n"
        "  --epsilon VALUE      covariance ridge (default 1e-3)\n"
        "  --threads N          CUDA threads/block; 0=auto, otherwise 32..224\n"
        "  --time-limit SEC     GPUBoscia time limit (default 3600)\n", program);
}

Options parse_options(int argc, char** argv) {
    Options o;
    auto value = [&](int& i) {
        if (++i >= argc) throw std::runtime_error("missing option value");
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--data") o.data = value(i);
        else if (arg == "--cardinality") o.cardinality = std::atoi(value(i));
        else if (arg == "--observations") o.observations = std::atoi(value(i));
        else if (arg == "--assets") o.assets = std::atoi(value(i));
        else if (arg == "--epsilon") o.epsilon = std::atof(value(i));
        else if (arg == "--threads") o.threads = std::atoi(value(i));
        else if (arg == "--time-limit") o.time_limit = std::atof(value(i));
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + arg);
    }
    return o;
}

// f(x) = 0.5 x'Hx - mu'x, grad f(x) = Hx - mu.
// H is the sample covariance plus epsilon I and is positive definite.
struct SparseSharpeObjective {
    static constexpr int n_vars = Variables;
    static constexpr int block_threads = Threads;
    static constexpr int shared_bytes = 16;
    const double* d_H;
    const double* d_mu;
    int assets;

    __device__ void prepare_block(int, unsigned char*) const {}

    __device__ void grad(int tid, const double* x, double* g,
                         unsigned char*) const {
        for (int i = tid; i < Variables; i += blockDim.x) {
            if (i >= assets) { g[i] = 0.0; continue; }
            double value = -d_mu[i];
            for (int j = 0; j < assets; ++j)
                value += d_H[i * assets + j] * x[j];
            g[i] = value;
        }
        __syncthreads();
    }

    __device__ double value_local(int tid, const double* x,
                                  unsigned char*) const {
        double value = 0.0;
        for (int i = tid; i < assets; i += blockDim.x) {
            value -= d_mu[i] * x[i];
            for (int j = 0; j < assets; ++j)
                value += 0.5 * x[i] * d_H[i * assets + j] * x[j];
        }
        return value;
    }
};

// Linear minimization oracle for
//   D = {(v,y): 0<=v_i<=U*y_i, 1'y<=m}
// intersected with the current branch-and-bound bounds on y.  Selecting an
// optional asset i has linear cost g_y[i] + min(0,U*g_v[i]); select the at-most
// m cheapest negative costs after honoring mandatory y entries.  Thus this
// oracle is an exact top-k operation and needs no nested LP/MIP solver.
struct LinkedCardinalityLMO {
    static constexpr bool supports_dicg = true;
    static constexpr int SortSize = 256;
    static_assert(SortSize >= MaxAssets && (SortSize & (SortSize - 1)) == 0,
                  "SortSize must be a power of two covering MaxAssets");
    int assets;
    int cardinality;
    const double* device_upper_bounds;
    const double* device_H;
    const double* device_mu;

    __device__ bool check_feasible_device(int tid, int, const double* lb,
        const double*, const uint8_t*, double*, double*) const {
        int mandatory = 0;
        for (int i = tid; i < assets; i += blockDim.x)
            mandatory += lb[MaxAssets + i] > 0.5;
        return feasibility_block_sum(mandatory) <= cardinality;
    }

    // Select mandatory assets, then the largest relaxed positions. The
    // selected direction is scaled to the quadratic minimum on the GPU.
    __device__ void integer_candidate_device(int tid, int n, const double* x,
        const double* lb, const double* ub, const uint8_t*, double* candidate) const {
        for (int j = tid; j < n; j += blockDim.x) candidate[j] = 0.0;
        __syncthreads();
        if (tid == 0) {
            int selected = 0;
            for (int i = 0; i < assets; ++i)
                if (lb[MaxAssets + i] > 0.5) {
                    candidate[MaxAssets + i] = 1.0;
                    ++selected;
                }
            while (selected < cardinality) {
                int best = -1;
                for (int i = 0; i < assets; ++i)
                    if (candidate[MaxAssets + i] == 0.0
                            && ub[MaxAssets + i] > 0.5 && x[i] > 1e-12
                            && (best < 0 || x[i] > x[best])) best = i;
                if (best < 0) break;
                candidate[MaxAssets + best] = 1.0;
                ++selected;
            }
            bool direction = false;
            for (int i = 0; i < assets; ++i)
                if (candidate[MaxAssets + i] == 1.0) {
                    candidate[i] = fmax(0.0, x[i]);
                    direction |= candidate[i] > 0.0;
                }
            if (!direction) {
                int best = -1;
                for (int i = 0; i < assets; ++i)
                    if ((candidate[MaxAssets + i] == 1.0
                            || (selected < cardinality && ub[MaxAssets + i] > 0.5))
                            && (best < 0 || device_mu[i] > device_mu[best])) best = i;
                if (best >= 0 && device_mu[best] > 0.0) {
                    candidate[MaxAssets + best] = 1.0;
                    candidate[best] = 1.0;
                }
            }
            double linear = 0.0, quadratic = 0.0, maximum = CUDART_INF;
            for (int i = 0; i < assets; ++i) {
                linear += device_mu[i] * candidate[i];
                if (candidate[i] > 0.0)
                    maximum = fmin(maximum, device_upper_bounds[i] / candidate[i]);
                for (int j = 0; j < assets; ++j)
                    quadratic += candidate[i] * device_H[i * assets + j] * candidate[j];
            }
            const double scale = linear > 0.0 && quadratic > 0.0
                ? fmin(linear / quadratic, maximum) : 0.0;
            for (int i = 0; i < assets; ++i) candidate[i] *= scale;
        }
        __syncthreads();
    }

    __device__ bool candidate_feasible_device(int tid, int n, const double* x,
        const double*, const double*, double tol) const {
        if (n != Variables) return false;
        double selected = 0.0;
        bool valid = true;
        for (int i = tid; i < assets; i += blockDim.x) {
            const double y = x[MaxAssets + i];
            if (x[i] < -tol || x[i] > device_upper_bounds[i] * y + tol)
                valid = false;
            selected += y;
        }
        const bool linked_bounds_valid = __syncthreads_and(valid);
        const double selected_sum = feasibility_block_sum(selected);
        return linked_bounds_valid && selected_sum <= cardinality + tol;
    }

    __device__ void compute_v(int tid, int, const double* grad,
                              const double* lb, const double* ub,
                              double* v) const {
        __shared__ double keys[SortSize];
        __shared__ int indices[SortSize];
        __shared__ int mandatory;

        if (tid == 0) mandatory = 0;
        __syncthreads();
        for (int i = tid; i < assets; i += blockDim.x)
            if (lb[MaxAssets + i] > 0.5) atomicAdd(&mandatory, 1);
        __syncthreads();

        // Sort optional assets once by their exact joint (v_i,y_i) cost.
        for (int i = tid; i < SortSize; i += blockDim.x) {
            const double cost = i < assets
                ? grad[MaxAssets + i]
                    + fmin(0.0, device_upper_bounds[i] * grad[i])
                : CUDART_INF;
            const bool eligible = i < assets && cost < 0.0
                && lb[MaxAssets + i] <= 0.5
                && ub[MaxAssets + i] > 0.5;
            keys[i] = eligible ? cost : CUDART_INF;
            indices[i] = eligible ? i : -1;
        }
        __syncthreads();
        for (int width = 2; width <= SortSize; width <<= 1)
            for (int stride = width >> 1; stride > 0; stride >>= 1) {
                for (int left = tid; left < SortSize; left += blockDim.x) {
                    const int other = left ^ stride;
                    if (other <= left) continue;
                    const bool ascending = (left & width) == 0;
                    const bool other_first = keys[other] < keys[left]
                        || (keys[other] == keys[left]
                            && indices[other] >= 0
                            && (indices[left] < 0 || indices[other] < indices[left]));
                    if (other_first == ascending) {
                        const double key = keys[left]; keys[left] = keys[other]; keys[other] = key;
                        const int index = indices[left]; indices[left] = indices[other]; indices[other] = index;
                    }
                }
                __syncthreads();
            }

        for (int i = tid; i < Variables; i += blockDim.x) v[i] = 0.0;
        __syncthreads();
        // check_feasible_device() prevents this path.  If bypassed, return zeros;
        // callers must treat that as an invalid oracle result for this node.
        if (mandatory > cardinality)
            return;
        const int optional_slots = cardinality - mandatory;
        for (int i = tid; i < assets; i += blockDim.x) {
            if (lb[MaxAssets + i] > 0.5) {
                v[MaxAssets + i] = 1.0;
                v[i] = grad[i] < 0.0 ? device_upper_bounds[i] : 0.0;
            }
        }
        for (int rank = tid; rank < optional_slots; rank += blockDim.x) {
            const int i = indices[rank];
            if (i >= 0) {
                v[MaxAssets + i] = 1.0;
                v[i] = grad[i] < 0.0 ? device_upper_bounds[i] : 0.0;
            }
        }
        __syncthreads();
    }

    __device__ void compute_inface_v(
        int tid, int, const double* grad, const double* x,
        const double* lb, const double* ub, double* a,
        double atol = 1e-6, double rtol = 1e-4) const {
        __shared__ double keys[SortSize];
        __shared__ int indices[SortSize];
        __shared__ unsigned char y_fixed[MaxAssets];
        __shared__ unsigned char v_zero[MaxAssets];
        __shared__ unsigned char link_active[MaxAssets];
        __shared__ int fixed_selected;
        __shared__ int force_full;

        if (tid == 0) { fixed_selected = 0; force_full = 0; }
        for (int i = tid; i < Variables; i += blockDim.x) a[i] = 0.0;
        __syncthreads();

        for (int i = tid; i < assets; i += blockDim.x) {
            const double U = device_upper_bounds[i];
            const double vi = x[i], yi = x[MaxAssets+i];
            const bool at_y_lower = fabs(yi - lb[MaxAssets+i])
                <= atol + rtol * fmax(fabs(yi), fabs(lb[MaxAssets+i]));
            const bool at_y_upper = fabs(yi - ub[MaxAssets+i])
                <= atol + rtol * fmax(fabs(yi), fabs(ub[MaxAssets+i]));
            y_fixed[i] = at_y_lower || at_y_upper;
            v_zero[i] = fabs(vi) <= atol + rtol * fabs(vi);
            link_active[i] = fabs(vi - U * yi)
                <= atol + rtol * fmax(fabs(vi), fabs(U * yi));

            if (y_fixed[i]) {
                const double fixed_y = at_y_lower
                    ? lb[MaxAssets+i] : ub[MaxAssets+i];
                a[MaxAssets+i] = fixed_y;
                if (fixed_y > 0.5) atomicAdd(&fixed_selected, 1);
                if (!v_zero[i]) {
                    if (link_active[i]) a[i] = U * fixed_y;
                    else a[i] = grad[i] < 0.0 ? U * fixed_y : 0.0;
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            double sum_y = 0.0;
            for (int i = 0; i < assets; ++i) sum_y += x[MaxAssets+i];
            force_full = fabs(sum_y - cardinality)
                <= atol + rtol * fmax(fabs(sum_y), (double)cardinality);
        }
        __syncthreads();

        for (int pos = tid; pos < SortSize; pos += blockDim.x) {
            bool eligible = false;
            double cost = CUDART_INF;
            if (pos < assets) {
                const double U = device_upper_bounds[pos];
                if (!y_fixed[pos]) {
                    cost = grad[MaxAssets+pos];
                    if (link_active[pos]) cost += U * grad[pos];
                    else if (!v_zero[pos]) cost += fmin(0.0, U * grad[pos]);
                    eligible = force_full || cost < 0.0;
                }
            }
            keys[pos] = eligible ? cost : CUDART_INF;
            indices[pos] = eligible ? pos : -1;
        }
        __syncthreads();
        for (int width = 2; width <= SortSize; width <<= 1)
            for (int stride = width >> 1; stride > 0; stride >>= 1) {
                for (int left = tid; left < SortSize; left += blockDim.x) {
                    const int right = left ^ stride;
                    if (right <= left) continue;
                    const bool ascending = (left & width) == 0;
                    const bool other_first = keys[right] < keys[left]
                        || (keys[right] == keys[left] && indices[right] >= 0
                            && (indices[left] < 0 || indices[right] < indices[left]));
                    if (other_first == ascending) {
                        const double key = keys[left]; keys[left] = keys[right]; keys[right] = key;
                        const int index = indices[left]; indices[left] = indices[right]; indices[right] = index;
                    }
                }
                __syncthreads();
            }
        const int slots = cardinality - fixed_selected;
        for (int rank = tid; rank < slots; rank += blockDim.x) {
            const int i = indices[rank];
            if (i >= 0) {
                a[MaxAssets+i] = 1.0;
                const double U = device_upper_bounds[i];
                if (link_active[i]) a[i] = U;
                else if (!v_zero[i]) a[i] = grad[i] < 0.0 ? U : 0.0;
            }
        }
        __syncthreads();
    }

    __device__ double dicg_maximum_step_local(
        int tid, int, const double* d, const double* x,
        const double* lb, const double* ub, double tol = 1e-9) const {
        double result = 1.0;
        for (int i = tid; i < assets; i += blockDim.x) {
            const double dv = d[i], dy = d[MaxAssets+i];
            if (dv > tol) result = fmin(result, x[i] / dv);
            const double link_d = dv - device_upper_bounds[i] * dy;
            const double link_x = x[i] - device_upper_bounds[i] * x[MaxAssets+i];
            if (link_d < -tol) result = fmin(result, link_x / link_d);
            if (dy > tol) result = fmin(result,
                (x[MaxAssets+i] - lb[MaxAssets+i]) / dy);
            else if (dy < -tol) result = fmin(result,
                (ub[MaxAssets+i] - x[MaxAssets+i]) / -dy);
        }
        return fmax(0.0, result);
    }

    __device__ double dicg_sum_maximum_step(
        int tid, int, const double* d, const double* x) const {
        double local_x = 0.0, local_d = 0.0;
        for (int i = tid; i < assets; i += blockDim.x) {
            local_x += x[MaxAssets+i];
            local_d += d[MaxAssets+i];
        }
        const double sum_x = lmo_block_sum(local_x, tid);
        const double sum_d = lmo_block_sum(local_d, tid);
        return sum_d < -1e-9
            ? fmax(0.0, fmin(1.0, (sum_x - cardinality) / sum_d)) : 1.0;
    }
};

} // namespace

int main(int argc, char** argv) {
    Options opts;
    try { opts = parse_options(argc, argv); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Argument error: %s\n", e.what());
        usage(argv[0]);
        return 2;
    }

    int total_rows = 0, available_assets = 0;
    std::vector<double> prices;
    try { prices = load_mat(opts.data, total_rows, available_assets); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Dataset error: %s\n", e.what());
        return 1;
    }
    if (opts.assets < 1 || opts.assets > MaxAssets
        || opts.assets > available_assets || opts.cardinality < 1
        || opts.cardinality > opts.assets || opts.epsilon <= 0.0
        || opts.threads < 0 || opts.threads > Threads
        || (opts.threads != 0 && opts.threads % 32 != 0)) {
        std::fprintf(stderr,
            "Invalid options: need 1 <= cardinality <= assets <= %d, "
            "assets <= %d available columns, and threads=0 or a multiple "
            "of 32 in [32,%d] (got %d).\n",
            MaxAssets, available_assets, Threads, opts.threads);
        return 2;
    }
    const int assets = opts.assets;
    if (assets < available_assets) {
        std::vector<double> selected((std::size_t)total_rows * assets);
        for (int t = 0; t < total_rows; ++t)
            for (int i = 0; i < assets; ++i)
                selected[(std::size_t)t * assets + i] =
                    prices[(std::size_t)t * available_assets + i];
        prices.swap(selected);
    }
    const int observations = opts.observations <= 0
        ? total_rows : std::min(opts.observations, total_rows);
    const int first = total_rows - observations;

    std::vector<double> returns((std::size_t)observations * assets);
    std::vector<double> mu(assets, 0.0);
    for (int t = 0; t < observations; ++t)
        for (int i = 0; i < assets; ++i) {
            const double r = prices[(std::size_t)(first + t) * assets + i] - 1.0;
            returns[(std::size_t)t * assets + i] = r;
            mu[i] += r / observations;
        }

    std::vector<double> H((std::size_t)assets * assets, 0.0);
    const double covariance_scale = observations > 1
        ? 1.0 / (observations - 1) : 1.0;
    for (int i = 0; i < assets; ++i)
        for (int j = 0; j < assets; ++j) {
            double covariance = 0.0;
            for (int t = 0; t < observations; ++t)
                covariance += (returns[(std::size_t)t * assets + i] - mu[i])
                            * (returns[(std::size_t)t * assets + j] - mu[j]);
            H[(std::size_t)i * assets + j] = covariance * covariance_scale
                + (i == j ? opts.epsilon : 0.0);
        }

    const std::vector<double> chol = cholesky_factor(H, assets);
    const std::vector<double> center = cholesky_solve(chol, mu, assets);
    const double ellipsoid_radius_sq = std::max(
        std::inner_product(mu.begin(), mu.end(), center.begin(), 0.0), 0.0);
    std::vector<double> upper_bounds(assets, 0.0);
    for (int i = 0; i < assets; ++i) {
        std::vector<double> unit(assets, 0.0);
        unit[i] = 1.0;
        const std::vector<double> inverse_column =
            cholesky_solve(chol, unit, assets);
        const double radius = std::sqrt(std::max(
            inverse_column[i] * ellipsoid_radius_sq, 0.0));
        upper_bounds[i] = std::max(center[i] + radius, 0.0);
    }

    double *d_H = nullptr, *d_mu = nullptr, *d_upper_bounds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_H, H.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, mu.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_upper_bounds, upper_bounds.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_H, H.data(), H.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu.data(), mu.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_upper_bounds, upper_bounds.data(),
        upper_bounds.size() * sizeof(double), cudaMemcpyHostToDevice));

    SparseSharpeObjective objective{
        d_H, d_mu, assets};
    Problem problem;
    problem.n_vars = Variables;
    problem.lb.assign(Variables, 0.0);
    problem.ub.assign(Variables, 0.0);
    // v is continuous; y is binary.  Inactive padded coordinates stay fixed 0.
    for (int i = 0; i < assets; ++i) {
        problem.ub[i] = upper_bounds[i];
        problem.ub[MaxAssets + i] = 1.0;
        problem.int_vars.push_back(MaxAssets + i);
    }

    LinkedCardinalityLMO lmo{
        assets, opts.cardinality, d_upper_bounds,
        d_H, d_mu};

    DeviceResidentConfig cfg;
    cfg.gpu_batch_size = 1024;
    cfg.num_streams = 8;
    cfg.gpu_block_threads = opts.threads;
    cfg.gpu_max_iters = 10000;
    cfg.integer_tol = 1e-5;
    cfg.abs_gap_tol = 1e-6;
    cfg.rel_gap_tol = 1e-2;
    auto settings = gpuboscia::create_default_settings();
    settings.device = cfg;
    settings.algorithm = FWAlgorithmKind::DICG;
    settings.line_search.kind = LineSearchKind::Secant;
    // The line-search tolerance must be tighter than the 1e-6 FW stopping
    // tolerance.  Otherwise secant can return gamma=0 while the node is still
    // unconverged, leaving every subsequent iteration at the identical point.
    settings.line_search.secant_iters = 16;
    settings.line_search.secant_tol = 1e-10;
    settings.line_search.secant_denom_eps = 1e-14;
    settings.time_limit_seconds = opts.time_limit;
    // Unconstrained quadratic minimum is a valid global lower bound.
    const double lower_bound = -0.5 * ellipsoid_radius_sq;
    settings.natural_lower_bound = lower_bound;

    const int launch_threads = opts.threads == 0
        ? std::min(Threads, ((Variables + 31) / 32) * 32) : opts.threads;
    std::printf("CUDA block threads: %d%s\n", launch_threads,
                opts.threads == 0 ? " (auto)" : "");

    Logger logger(true, 2000);
    settings.progress = [&logger](int i, int o, double b,
                                  double incumbent, bool fresh) {
        logger.print_row(i, o, b, incumbent, fresh);
    };
    LoggerParams lp;
    lp.traversal_strategy = "best_first (shared queue)";
    lp.branching_strategy = "branch_pow2(k=3, binary)";
    lp.algorithm = fw_algorithm_name(settings.algorithm);
    lp.line_search_method = line_search_name(settings.line_search.kind);
    lp.n_vars = Variables;
    lp.n_int_vars = assets;
    lp.abs_gap_tol = cfg.abs_gap_tol;
    lp.rel_gap_tol = cfg.rel_gap_tol;
    lp.time_limit_s = opts.time_limit;
    lp.natural_lower_bound = lower_bound;
    logger.print_header(lp);

    const gpuboscia::Result result = gpuboscia::solve(
        problem, objective, lmo, settings);
    SolutionStats stats;
    stats.primal_objective = result.objective;
    stats.dual_bound = result.dual_bound;
    stats.total_nodes = result.nodes_processed;
    stats.total_time_s = result.elapsed_seconds;
    stats.solution = result.x;
    stats.solution_status = result.reached_time_limit ? "Time limit reached"
        : (result.reached_tolerance ? "Reached tolerance" : "Finished");
    logger.print_footer(stats);

    int selected = 0, selected_y = 0;
    double scale = 0.0;
    if (result.x.size() == Variables) {
        for (int i = 0; i < assets; ++i)
            if (result.x[i] > 1e-7) {
                ++selected;
                scale += result.x[i];
            }
        for (int i = 0; i < assets; ++i)
            selected_y += result.x[MaxAssets + i] > 0.5;
    }
    std::printf("\nContinuous-weight sparse Sharpe portfolio:\n"
                "  Dataset:      %s\n"
                "  Window:       last %d of %d observations\n"
                "  Candidates:   first %d of %d asset columns\n"
                "  Portfolio:    optimize at most %d candidates\n"
                "  Bounds U:     [%.10g, %.10g]\n"
                "  Support/y:    %d/%d\n"
                "  Portfolio weights:\n",
                opts.data.c_str(), observations, total_rows, assets,
                available_assets, opts.cardinality,
                *std::min_element(upper_bounds.begin(), upper_bounds.end()),
                *std::max_element(upper_bounds.begin(), upper_bounds.end()),
                selected, selected_y);
    if (selected == 0 || !(scale > 0.0) || result.x.size() != Variables) {
        std::printf("    no incumbent portfolio found\n");
    } else {
        double p_dot_v = 0.0, v_h_v = 0.0;
        for (int i = 0; i < assets; ++i) {
            p_dot_v += mu[i] * result.x[i];
            for (int j = 0; j < assets; ++j)
                v_h_v += result.x[i] * H[(std::size_t)i * assets + j]
                       * result.x[j];
        }
        const double recomputed_objective = 0.5 * v_h_v - p_dot_v;
        const double regularized_sharpe_v = v_h_v > 0.0
            ? p_dot_v / std::sqrt(v_h_v)
            : std::numeric_limits<double>::infinity();
        const double implied_regularized_sharpe = recomputed_objective < 0.0
            ? std::sqrt(-2.0 * recomputed_objective) : 0.0;

        double expected = 0.0, raw_variance = 0.0;
        for (int i = 0; i < assets; ++i)
            if (result.x[i] > 1e-7) {
                const double weight = result.x[i] / scale;
                std::printf("    asset %-3d v=%.10g  w=%.10g  y=%.0f\n", i + 1,
                            result.x[i], weight, result.x[MaxAssets + i]);
                expected += mu[i] * weight;
                for (int j = 0; j < assets; ++j)
                    if (result.x[j] > 1e-7)
                        raw_variance += weight * (result.x[j] / scale)
                            * (H[(std::size_t)i * assets + j]
                               - (i == j ? opts.epsilon : 0.0));
            }
        const double regularized_variance = v_h_v / (scale * scale);
        const double raw_sharpe = raw_variance > 0.0
            ? expected / std::sqrt(raw_variance)
            : std::numeric_limits<double>::infinity();
        std::printf(
            "  Transformed-problem diagnostics:\n"
            "    p^T v:                       %.12g\n"
            "    v^T Q_epsilon v:             %.12g\n"
            "    0.5 v^T Q_epsilon v-p^T v:  %.12g\n"
            "    Solver objective difference: %.3e\n"
            "    Scale stationarity residual:  %.3e\n"
            "    Regularized Sharpe from v:    %.12g\n"
            "    sqrt(-2 F(v)):                %.12g\n"
            "  Expected return:                %.10g\n"
            "  Raw volatility (Sigma):         %.10g\n"
            "  Regularized volatility (Q_eps): %.10g\n"
            "  Raw Sharpe ratio:                %.10g\n"
            "  Regularized Sharpe ratio:        %.10g\n",
            p_dot_v, v_h_v, recomputed_objective,
            recomputed_objective - result.objective,
            p_dot_v - v_h_v, regularized_sharpe_v,
            implied_regularized_sharpe, expected,
            std::sqrt(std::max(raw_variance, 0.0)),
            std::sqrt(std::max(regularized_variance, 0.0)),
            raw_sharpe, regularized_sharpe_v);
    }

    cudaFree(d_H);
    cudaFree(d_mu);
    cudaFree(d_upper_bounds);
    return result.x.empty() ? 1 : 0;
}
