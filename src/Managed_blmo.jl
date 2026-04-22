# mutable struct ManagedLMO{LMO<:FrankWolfe.LinearMinimizationOracle} <:
#                FrankWolfe.LinearMinimizationOracle
#     lmo::LMO
#     lower_bounds::Vector{Float64}
#     upper_bounds::Vector{Float64}
#     int_vars::Vector{Int}
#     n::Int
#     solving_time::Float64
# end

Adapt.@adapt_structure Boscia.ManagedLMO