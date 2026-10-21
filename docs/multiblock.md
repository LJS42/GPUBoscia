# Persistent multi-block FW

`DeviceResidentHybridSolver` still defaults to one block per B&B node. CPU
selection, branching, pruning, and batch scheduling are unchanged. MultiBlock
uses a cooperative launch containing exactly `N*K` blocks and one persistent
GPU solve per node. It never launches a kernel or asks the CPU to advance an
FW iteration. No Cluster backend is provided.

## Enable

```cpp
DeviceResidentConfig cfg;
cfg.execution_mode = FWExecutionMode::MultiBlock;
cfg.blocks_per_node = 4;
cfg.gpu_batch_size = 1024; // Requested upper bound, capped by residency.
DeviceResidentHybridSolver<MyObjective, FrankWolfeSecant, MyLMO> solver(
    problem, objective, algorithm, branch_strategy, cfg, lmo);
```

`FWExecutionMode::SingleBlock` remains the default. New fields are appended to
`DeviceResidentConfig`, preserving earlier positional aggregate initializers.
`solver.effective_gpu_batch_size()` and `solver.effective_num_streams()` expose
the effective configuration. MultiBlock uses exactly one stream; SingleBlock
retains existing stream behavior and the optional paired DICG path.

MultiBlock requires CUDA Toolkit **11.6 or newer** for `cuda::atomic_ref`, a
cooperative-launch-capable GPU, a valid warp-aligned block size, and this
optional const Objective method:

```cpp
__device__ void grad_multiblock(
    MultiBlockContext& ctx, const double* x, double* gradient,
    unsigned char* workspace) const;
```

Existing `grad`, `value_local`, and `prepare_block` remain required and keep
their signatures. LMOs need no new interface. The following optional **host**
method requests additional per-node global Objective scratch:

```cpp
size_t multiblock_workspace_bytes(int blocks_per_node, int threads_per_block) const;
```

The solver allocates the maximum of that result, `Objective::shared_bytes`,
and 16 bytes, rounded up to 16-byte alignment. Without the hook,
`shared_bytes` must already cover every access made by both Objective paths.
The Objective owns its scratch layout and any additional problem-data
allocation; the solver cannot infer space for an arbitrary CUDA callback.
Invalid sizes that overflow allocation arithmetic are rejected; CUDA
allocation failures report the CUDA error. Workspace is **global**, despite
the historical `shared_bytes` name.

An explicit request throws instead of falling back when the toolkit,
Objective interface, device capability, block size, shared memory, or complete
node-group residency is unsupported. `blocks_per_node` must be positive;
K=1 is legal. Forced `DICGOracleParallelism::Paired` cannot be combined with
MultiBlock. DICG itself is supported with its leader-block oracles.

## Objective synchronization contract

The context exposes `node_id`, `block_rank`, `block_count`, `thread_rank`,
`threads_per_block`, `global_tid`, and `total_threads`, plus `block_sync()`
and `node_sync()`. Coordinate loops usually use:

```cpp
for (int j = ctx.global_tid; j < n_vars; j += ctx.total_threads)
    gradient[j] = /* Objective-defined expression */;
```

Nonseparable Objectives may perform multiple node collectives within
`grad_multiblock()`. See `SumObjective` in `tests/test_multiblock.cu`: it first
reduces within each block, stores one partial per block, synchronizes the node,
combines the partials, writes disjoint gradient partitions, and synchronizes
again. No separability or quadratic structure is required by the backend.

Every thread in all K blocks must execute matching node collectives in the
same order. Branching around a collective, early return by only some threads,
or `this_grid().sync()` inside an Objective violates the contract. All
cross-block scratch writes must have one owner or an explicit atomic
protocol, and dependent reads need a matching node barrier. User-authored
invalid CUDA synchronization is outside the solver's guarantee.

`prepare_block()` runs on the leader block once, before the first gradient
request. Shared per-node preparation writes are published by the dispatch
barrier. If helpers need block-local initialization, do it inside
`grad_multiblock()`. All callbacks must respect their actual participant set;
legacy callbacks see physical `blockIdx.x = node_id*K`, not `node_id`.
Objective state should be kernel-copyable as before. Composition in the
adapter also supports Objective classes declared `final`.

A multi-block gradient must compute the same full gradient as `grad`, and
maintain any Objective workspace/cache contract used by `value_local`,
`value_from_factor_local`, or `quadratic_curvature_local`. Optional cached-value
and quadratic-curvature hooks are forwarded by the adapter; they are never
assumed to exist or required for general Objectives.

## FW phases and storage

Both entry kernels call the same device FW body. The leader owns existing
block-shared scalars, stopping decisions, line search, feasibility checks,
candidate scoring, and final certificate recomputation. A per-node GPU command
record dispatches gradient calls (including every secant trial gradient), gap
reductions, optional forward LMO calls, and vector updates to the helpers. Helpers wait at **their own
node's** next dispatch barrier while the leader runs other phases. A final
stop command releases the entire group, including when feasibility fails
before the first gradient. Different nodes can exit at different iterations.

LMOs can opt in with a const device method:

```cpp
void compute_v_multiblock(MultiBlockContext& ctx, int n,
    const double* gradient, const double* lb, const double* ub, double* v) const;
```

Every thread of all K blocks calls this method. It must produce the same vertex
as `compute_v`; matching internal node barriers are permitted. The caller
publishes inputs and synchronizes output with node barriers. `BoxLMO` implements
this hook using disjoint global-thread-stride partitions. `UnitSimplexLMO`
(and `EqualitySimplexLMO`) sorts 128-element runs in shared memory, merges
runs with one node barrier per merge level, and scans capacities in shared
memory with a small global array of tile totals. For n=257 this requires four
internal node barriers instead of a node barrier at every comparator/scan
stage. Networks of at most 128 entries stay on the leader block.

`BirkhoffLMO` runs the block-local Hungarian algorithm on the leader block,
including within MultiBlock FW. It has no multi-block oracle callback or
per-node global LMO scratch. Gradient, gap and vector-update collaboration
remain available for objectives used with Birkhoff.

An LMO may declare `static constexpr size_t multiblock_workspace_bytes`.
The solver allocates that many bytes per node, rounded up to 16-byte alignment,
and exposes them as `ctx.lmo_workspace`. Scratch is independent for each node,
reused across calls, and must be initialized by the callback before reading it.
This is separate from Objective scratch. Simplex uses O(MaxN) global scratch;
Birkhoff, legacy LMOs and Box need none. A derived LMO overriding `compute_v` must also
supply a matching collective or hide the inherited hook to retain fallback.
Legacy LMOs execute on all threads of the leader block, including LMOs
containing `__syncthreads()`. DICG in-face work remains on that leader. Gap
reductions write one separate partial per block and the leader combines them;
vector updates use disjoint global-thread-stride partitions. Step size and
stop criteria retain their existing policy implementations, iteration
continuation semantics, and double precision. Parallel summation changes
floating-point reduction order, so near-threshold iteration counts can differ
on general inputs. Tests use documented absolute/relative tolerances rather
than promising bitwise reproducibility.

There is one shared set of node vectors, not K full copies. For batch size B,
the solver's main **device** vectors use approximately:

| Dimension n | FW open-loop, 48*B*n bytes | FW secant, 64*B*n bytes |
|---|---:|---:|
| 100,000 | 4.8 MB per node | 6.4 MB per node |
| 1,000,000 | 48 MB per node | 64 MB per node |
| 10,000,000 | 480 MB per node | 640 MB per node |

These decimal MB figures include x/lb/ub/best candidate plus algorithm vectors.
DICG adds two vectors (16*B*n bytes). Add the Objective's and LMO's per-node scratch,
one n-byte integer-flag vector per stream, small output arrays, and
`B*(sizeof(DRSNodeControl)+K*sizeof(double))` control/reduction storage.
Pinned host node buffers add approximately `32*B*n` bytes, with additional
host queue, completed-batch, and branching vectors. Large dimensions require
choosing a memory-appropriate requested batch even if occupancy admits more.

## Barrier and launch safety

Each node has separate aligned plain unsigned arrival/generation storage,
initialized using `cudaMemsetAsync` on the launch stream. Device-scoped
`cuda::atomic_ref` accesses wrap that storage; no raw host writes construct or
initialize CUDA atomic objects.

Before arrival, a block barrier publishes every block thread's writes to its
representative. One thread per block performs an acquire-release arrival RMW.
The RMW chain transitively publishes all arriving blocks' writes to the last
arrival, which resets arrivals and release-stores the incremented generation.
Other representatives acquire-load the generation, then a block barrier
makes acquired writes visible to their entire blocks. A block cannot finish
the next generation without all other blocks, so a delayed waiter cannot be
lapped by a generation wraparound. No legacy volatile-only publication is used.

The host queries cooperative support and occupancy of the **actual** kernel,
block size, and dynamic shared memory, including static shared-memory and
opt-in limits. It computes:

```
resident_blocks = active_blocks_per_sm * SM_count
max_nodes = min(resident_blocks, max_grid_x) / K
effective_batch = min(requested_batch, max_nodes)
```

Only `cudaLaunchCooperativeKernel` launches MultiBlock. The grid has complete
groups and fits residency. Stream slots and barrier storage are reused only
after completion. The launch worker selects the same CUDA device used for
occupancy and allocation, since CUDA current device is host-thread-local.
The new backend contains no grid-wide phase barriers. The pre-existing paired
DICG backend remains separate and still uses its original grid synchronization.

## Run on an A40

This development host has **no nvcc, NVIDIA driver tools, or GPU**, so CUDA
compilation, GPU execution, sanitizers, and performance have not been run here.
CPU-only Release build, CPU tests, header checks, and package consumer tests
passed. GPU validation remains required; there are no speedup claims.

```bash
nvidia-smi
nvcc --version
compute-sanitizer --version
cmake -S . -B build-a40 -DCMAKE_BUILD_TYPE=Release \
  -DGPUBOSCIA_CUDA_ARCH=86 -DGPUBOSCIA_BUILD_EXAMPLES=OFF \
  -DGPUBOSCIA_BUILD_BENCHMARKS=ON
cmake --build build-a40 -j
ctest --test-dir build-a40 --output-on-failure

compute-sanitizer --error-exitcode 1 --tool memcheck ./build-a40/test_multiblock
compute-sanitizer --error-exitcode 1 --tool initcheck ./build-a40/test_multiblock
compute-sanitizer --error-exitcode 1 --tool racecheck ./build-a40/test_multiblock
compute-sanitizer --error-exitcode 1 --tool synccheck ./build-a40/test_multiblock
```

`multiblock_tests` exercises K=1/2/4/8 where occupancy permits, at least 40,000
barrier generations per node, publication of every thread's writes and
cross-block reductions, internal Objective
barriers, and unequal node lifetimes. Slow nodes intentionally wait until a
fast node has completed 100 iterations **before entering their first node
barrier**, exposing accidental grid-wide synchronization. Numerical comparisons
cover open-loop FW, secant, quadratic-exact, DICG, early convergence,
iteration continuation, different FW iteration counts in the same grid,
iteration limit, and mixed feasible/infeasible groups. The 1024-variable
fixture gives all eight blocks actual gradient work at K=8. The custom LMO has block collectives and checks
that only complete leaders call it. A `final` Objective and dynamic Objective
workspace hook are exercised. Configuration tests cover missing callbacks,
invalid K/block size, unsupported cooperative capability, capacity rounding,
full boundary batches, and partial tail batches. Kernel comparisons require
iterate error <= `1e-10*(1+abs(reference))` and value/gap error <=
`1e-9*(1+abs(reference))`, and identical statuses/iteration counts on these
fixtures. End-to-end integration additionally checks the analytic optimum.

Tests return 77 if no CUDA device is available or the toolkit lacks atomic_ref.
K values lacking residency are explicitly reported as skipped. On a device
without cooperative launch, the rejection path is tested before returning 77.
CTest gives the stress test a 300-second timeout; sanitizer runs can take much
longer and can be run directly as above. Also run the existing pipeline under
sanitizers when time permits.

Sanitizer reports must be retained with CUDA/driver/GPU versions. Racecheck
primarily checks shared-memory hazards; it does not prove this global-memory
atomic protocol. Synccheck does not prove matching custom node generations.
A clean sanitizer run complements, rather than replaces, visibility and
independence tests. Do not classify a report as a tooling limitation without
examining it.

## Benchmark

```bash
./build-a40/benchmark_multiblock > multiblock.csv
# Optional n=10^7; requires substantial device and host memory:
./build-a40/benchmark_multiblock --large > multiblock-large.csv
```

The synthetic nonseparable Objective is used only by the benchmark. It tests
the following dimensions and blocks per node, with a separate SingleBlock
baseline for every dimension and the same requested batch and stream count:

| n | MultiBlock K |
|---|---|
| 5,000 | 1, 2, 4, 8 |
| 10,000 | 1, 2, 4, 8 |
| 50,000 | 1, 2, 4, 8, 16 |
| 100,000 | 1, 2, 4, 8, 16 |
| 1,000,000 | 1, 2, 4, 8, 16 |

`--large` additionally tests n=10,000,000 with K=1/2/4/8/16. These are
whole-relaxation and B&B benchmarks using BoxLMO; they do not measure large
Simplex or Birkhoff oracles. Small cases in `benchmark_lmo` are correctness
and overhead controls. The current fixed-size Simplex oracle stores integer
flags in the kernel argument and block-local sorting arrays in shared memory;
simply increasing MaxN to these sizes is not supported by that implementation.
Birkhoff's variable count is Order squared, with a different scaling cost.

CSV records actual-kernel occupancy,
cooperative capacity, effective batch, device allocation delta, whole single
relaxation latency, whole B&B elapsed time and node throughput, and standalone
gradient/LMO/barrier microbenchmarks. Whole solves include transfers and CPU
scheduling/postprocessing but exclude solver construction; single-node latency
averages two runs after one warmup. B&B uses four integer coordinates and a
30-second time limit and reports whether the limit was reached. Do not compare
an incomplete run as if it completed the same search.

Stage timings are separate persistent microbenchmarks, include dependent
barriers, and estimate microseconds using `clock64` and the nominal device
clock rate; frequency changes affect their accuracy. They are not per-phase
profiling of the B&B kernel. Device-memory deltas use `cudaMemGetInfo`, exclude
host buffers, and can be perturbed by other processes. For accurate full-kernel
timelines and bandwidth, collect Nsight traces on an otherwise idle A40 and
record driver/toolkit, clocks, configuration, and output CSV. No measurements
are supplied from this CPU-only host.

NVIDIA references: [cooperative launch residency limits](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
and [libcu++ synchronization primitive availability](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html).

### Other LMO microbenchmarks

```bash
cmake --build build-a40 --target test_multiblock test_multiblock_lmo benchmark_lmo -j
./build-a40/test_multiblock
./build-a40/test_multiblock_lmo --test
./build-a40/benchmark_lmo > lmo.csv
# Larger sorting / assignment problems; Hungarian can take much longer:
./build-a40/benchmark_lmo --large > lmo-large.csv
```

The oracle benchmark checks each vertex against CPU and SingleBlock before
reporting CUDA-event timings for SingleBlock, leader-only cooperative execution
(K=2/4/8), and MultiBlock K=1/2/4/8. `leader` is a control with the same
cooperative group and caller barrier but the original block-local oracle;
compare `multi` against `leader` at the same K to isolate oracle changes.
`lmo_blocks` reports the nominal number of blocks owning tiles or columns
(the remaining blocks still join node barriers). `threads` records block size.
`reserved_scratch_bytes_per_node` is the allocated global scratch size, also
reserved for the controls that do not use it.
Three independent nodes have different inputs. Fixtures include tied costs,
integer capacities, lower bounds, equality constraints, forbidden assignment
edges, forced assignments and an infeasible assignment. `batch_us` is the
median of five samples for the **entire three-node batch**, with repeated
oracle calls per launch, including dependent barriers and amortized launch
overhead. Compare the same LMO and n; integer-only fixtures have separate
labels. This is an oracle microbenchmark, not end-to-end B&B timing.

Small sorting networks can be slower across blocks: node barriers and global
scratch may outweigh useful parallel work. The `--large` fixtures include
simplex MaxN=1025 and block-local Birkhoff Order=256. No speedup is assumed
before GPU measurements. DICG in-face oracles remain block-local.
Default fixtures include Birkhoff Order=129 as a leader-fallback regression
check; all Birkhoff rows report `lmo_blocks=1`. Test mode also uses 32- and
96-thread blocks. Simplex tests check barrier generations as well as vertices.
For K=1 the node barrier reduces to `__syncthreads()`, and Simplex reuses its
original shared-memory oracle.
