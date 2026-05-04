function gpu_start_point_initialize(
    lmo::Boscia.TimeTrackingLMO,
    active_set::FrankWolfe.ActiveSet,
    pre_computed_set,
)
    if lmo.ncalls == 0
        return Float64.(FrankWolfe.get_active_set_iterate(active_set))
    end

    if pre_computed_set === nothing || isempty(pre_computed_set)
        x0 = Float64.(FrankWolfe.get_active_set_iterate(active_set))
    else
        x0 = sum(pre_computed_set) / length(pre_computed_set)
    end

    lb = lmo.lmo.lower_bounds
    ub = lmo.lmo.upper_bounds
    int_vars = lmo.lmo.lmo.int_vars
    for (idx, i) in enumerate(int_vars)
        x0[i] = clamp(x0[i], lb[idx], ub[idx])
    end

    return x0
end