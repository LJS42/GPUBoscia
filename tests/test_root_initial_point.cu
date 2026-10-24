#include <gpuboscia/gpuboscia.cuh>
#include <iostream>

struct ObservedQuadratic {
    static constexpr int n_vars=2,block_threads=128,shared_bytes=0;
    double* first;
    unsigned* calls;
    __device__ void prepare_block(int,unsigned char*) const {}
    __device__ void record(const double* x) const {
        if(atomicAdd(calls,1u)==0) {first[0]=x[0];first[1]=x[1];}
    }
    __device__ void grad(int tid,const double* x,double* g,unsigned char*) const {
        if(tid==0) record(x);
        for(int j=tid;j<2;j+=blockDim.x) g[j]=2*(x[j]-(j ? .75 : .25));
        __syncthreads();
    }
    __device__ void grad_multiblock(MultiBlockContext& ctx,const double* x,double* g,unsigned char*) const {
        if(ctx.global_tid==0) record(x);
        for(int j=ctx.global_tid;j<2;j+=ctx.total_threads) g[j]=2*(x[j]-(j ? .75 : .25));
        ctx.node_sync();
    }
    __device__ double value_local(int tid,const double* x,unsigned char*) const {
        double value=0;
        for(int j=tid;j<2;j+=blockDim.x) {double d=x[j]-(j ? .75 : .25);value+=d*d;}
        return value;
    }
};

int main() {
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
    try {
        double* first=nullptr;unsigned* calls=nullptr;
        DRS_CUDA_CHECK(cudaMalloc(&first,2*sizeof(double)));
        DRS_CUDA_CHECK(cudaMalloc(&calls,sizeof(unsigned)));
        Problem problem{2,{0,0},{1,1},{0,1}};
        cudaDeviceProp props{};int device=0;
        DRS_CUDA_CHECK(cudaGetDevice(&device));
        DRS_CUDA_CHECK(cudaGetDeviceProperties(&props,device));
        for(int mode:{0,1,2}) {
            if(mode && !props.cooperativeLaunch) continue;
            if(mode==2 && !DRS_HAS_MULTIBLOCK_ATOMICS) continue;
            for(bool supplied:{false,true}) {
                DRS_CUDA_CHECK(cudaMemset(calls,0,sizeof(unsigned)));
                auto settings=gpuboscia::create_default_settings();
                settings.device.gpu_batch_size=1;
                settings.device.num_streams=1;
                settings.device.gpu_max_iters=1;
                settings.device.gpu_block_threads=128;
                settings.device.execution_mode=mode==2 ? FWExecutionMode::MultiBlock : FWExecutionMode::SingleBlock;
                settings.device.blocks_per_node=3;
                using Parallel=DeviceResidentConfig::DICGOracleParallelism;
                settings.device.dicg_oracle_parallelism=mode==1 ? Parallel::Paired : Parallel::Serial;
                settings.branch=[](const Node&,int&,const std::vector<double>&,const std::vector<int>&) {
                    return std::vector<Node>{};
                };
                if(supplied) settings.root_initial_point={.25,.75};
                auto result=gpuboscia::solve_with_algorithm(problem,ObservedQuadratic{first,calls},
                    BoxLMO{},settings,DICGOpenLoop{});
                double observed[2];
                DRS_CUDA_CHECK(cudaMemcpy(observed,first,sizeof(observed),cudaMemcpyDeviceToHost));
                if(observed[0]!=(supplied ? .25 : 1.) || observed[1]!=(supplied ? .75 : 1.))
                    throw std::runtime_error("first objective call did not receive the expected root iterate");
                if(result.nodes_processed!=1)
                    throw std::runtime_error("unexpected root processing count");
                std::cout<<"mode="<<mode<<" supplied="<<supplied<<" passed\n";
            }
        }
        DRS_CUDA_CHECK(cudaFree(first));DRS_CUDA_CHECK(cudaFree(calls));
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
