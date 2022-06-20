"""
    find_initial_minima(geometry::Geometry, ϵ::Real, s::Integer, Nsols::Integer=1)

Find the `Nsols` initial minima of a loss function that connect the source and observer.
"""
function find_initial_minima(geometry::Geometry, ϵ::Real, s::Integer, Nsols::Integer=1)
    loss = setup_initial_loss(geometry, ϵ, s)
    X = [find_initial_minimum(loss, geometry)]

    if isnothing(X[1])
        return fill!(zeros(geometry.dtype, 7), NaN)
    end

    for i in 2:Nsols
        Xnew = find_initial_minimum(p -> loss(p, X), geometry)
        # Terminate the search
        if isnothing(Xnew)
            @info "Initial search terminated with $(i-1)/$Nsols solutions."
            push!(X, fill!(zeros(geometry.dtype, 7), NaN))
            break
        end
        push!(X, Xnew)
    end
    # Return and turn this into a matrix
    return mapreduce(permutedims, vcat, X)
end


"""
    find_initial_minimum(loss::Function, geometry::Geometry)

Find the initial minimum of a loss function that connects the source and observer.
"""
function find_initial_minimum(loss::Function, geometry::Geometry)
    @assert ~(geometry.direction_coords in shadow_coords) "Shadow minimum finder not supported."

    @unpack Ninit, loss_atol, alg, optim_options = geometry.opt_options
    for i in 1:Ninit
        opt = optimize(loss, rvs_sphere(dtype=geometry.dtype), alg, optim_options)
        if isapprox(opt.minimum, 0.0, atol=loss_atol)
            push!(opt.minimizer, geometry.arrival_time, geometry.redshift, opt.minimum, geometry.nloops, geometry.ϕkilling)
            return opt.minimizer
        end
    end

    return nothing
end


"""
    getθmax(relθmax::Real, ϵ::Real, ϵ0::Real,  nloops::Real)

Get the search radius as a function of ϵ and nloops. This is an empirical relation.
"""
function getθmax(relθmax::Real, ϵ::Real, ϵ0::Real,  nloops::Real)
    return relθmax * (ϵ > 0 ? ϵ : ϵ0) / sqrt(nloops + 0.25)
end


"""
    find_consecutive_minimum(
        geometry::Geometry,
        ϵ::Real,
        s::Integer,
        prev_init_direction::Vector{<:Real},
        ϵ0::Real,
        nloops::Real,
        prevϕkill::Real
    )

Find a trajectory to the observer (GSHE or geodesic) that is sufficiently close to previous
initial direction.
"""
function find_consecutive_minimum(
    geometry::Geometry,
    ϵ::Real,
    s::Integer,
    prev_init_direction::Vector{<:Real},
    ϵ0::Real,
    nloops::Real,
)
    @assert ~(geometry.direction_coords in shadow_coords) "Shadow minimum finder not supported."

    if any(isnan.(prev_init_direction))
        @warn "Skipping as `prev_init_direction` contains NaNs."
        return fill!(Vector{geometry.dtype}(undef, length(prev_init_direction) + 5), NaN)
    end

    @unpack alg, optim_options, relθmax, loss_atol, Nconsec, gshe_convergence_verbose = geometry.opt_options
    θmax = getθmax(relθmax, ϵ, ϵ0, nloops)
    loss = setup_consecutive_loss(geometry, ϵ, s, nloops)
    min_loss = Inf64
    for i in 1:Nconsec
        # Sample initial position and inv transform it
        p0 = rvs_sphere_y(θmax; dtype=geometry.dtype)
        @. p0 = atan_invtransform(p0, θmax)

        opt = optimize(p -> loss(p, prev_init_direction, θmax), p0, alg, optim_options)
        # Update the minimum loss
        if opt.minimum < min_loss
            min_loss = opt.minimum
        end
        if isapprox(opt.minimum, 0.0, atol=loss_atol)
            # Transform back to the default coordinate system
            @. opt.minimizer = atan_transform(opt.minimizer, θmax)
            rotate_from_y!(opt.minimizer, prev_init_direction)
            push!(opt.minimizer, geometry.arrival_time, geometry.redshift, opt.minimum, geometry.nloops, geometry.ϕkilling)

            return opt.minimizer
        else
            # Bump up the search radius but keep it restricted to some max value.
            if θmax < 0.15π * ((ϵ > 0 ? ϵ : ϵ0) / 0.1)
                θmax *= 2
            end
        end
    end

    if gshe_convergence_verbose
        @warn "GSHE search terminated with no solution found."
        flush(stdout)
    end

    # Return a vector of NaNs and put the min loss there as well
    out = fill!(Vector{geometry.dtype}(undef, length(prev_init_direction) + 5), NaN)
    out[5] = min_loss
    return out
end
