module Boscia_FWonGPU

import Adapt
using CUDA
#CUDA.set_runtime_version!(v"12.6")
using FrankWolfe
using Boscia

include("./BLMO.jl")
include("./Managed_blmo.jl")
include("./StandardFrankWolfe.jl")

end # module FrankWolfe_GPU
