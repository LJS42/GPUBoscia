module Boscia_FWonGPU

import Adapt
using CUDA
#CUDA.set_runtime_version!(v"12.6")
using FrankWolfe
using Boscia
using Bonobo

include("./Branching_nodes.jl")
include("./BLMO.jl")
include("./Managed_blmo.jl")
include("./Initialize.jl")
include("./StandardFrankWolfe.jl")
include("./DICG.jl")

end # module FrankWolfe_GPU
