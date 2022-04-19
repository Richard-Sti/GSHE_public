"""
    find_geodesic_minima(geometry::Geometry, Nsols::Integer=1)

Find `Nsols` unique minima of a geodesic for a given geometry.
"""
function find_geodesic_minima(geometry::Geometry, Nsols::Integer=1)
    loss = setup_geodesic_loss(geometry)
    X = [find_geodesic_minimum(loss, geometry)]

    for i in 2:Nsols
        Xnew = find_geodesic_minimum(p -> loss(p, X), geometry)
        # Terminate the search
        if Xnew === nothing
            @info "Search terminated with $(i-1)/$Nsols solutions."
            break
        end
        push!(X, Xnew)
    end
    # Return and turn this into a matrix
    return  mapreduce(permutedims, vcat, X)
end


"""
    find_geodesic_minimum(loss::Function, geometry::Geometry)

Find the minimum of a geodesic `loss` on the surface of a sphere. Attempts many tries until
a desired loss is found. Starts eachs attempt with randomly sampled position on the sky.
"""
function find_geodesic_minimum(loss::Function, geometry::Geometry)
    @unpack Nattempts_geo, loss_atol, alg, optim_options = geometry.opt_options
    for __ in 1:Nattempts_geo
        opt = optimize(loss, rvs_sphere(dtype=geometry.dtype), alg, optim_options)
        if isapprox(opt.minimum, 0.0, atol=loss_atol)
            push!(opt.minimizer, geometry.arrival_time, geometry.redshift)
            return opt.minimizer
        end
    end

    return nothing
end


"""
    θmax_scaling(θmax0::Real, ϵ::Real)

Calculate :math:`θmax = θmax0 + 0.5√ϵ`, however maximum value is capped at π/4.
"""
function θmax_scaling(θmax0::Real, ϵ::Real)
    θmax =  θmax0 + sqrt(ϵ) / 2
    θmax > π / 4 ? (return π/4) : return θmax
end


"""
    find_restricted_minimum(
        geometry::Geometry,
        ϵ::Real,
        s::Integer,
        prev_init_direction::Vector{<:Real}
    )

Find a GSHE trajectory minimum near some previous initial direction.
"""
function find_restricted_minimum(
    geometry::Geometry,
    ϵ::Real,
    s::Integer,
    prev_init_direction::Vector{<:Real}
)
    @unpack alg, optim_options, θmax0, loss_atol, Nattempts_gshe = geometry.opt_options
    loss = setup_gshe_loss(geometry, ϵ, s)
    for i in 1:Nattempts_gshe
        θmax = θmax_scaling(θmax0, ϵ)
        # Sample initial position and inv transform it
        p0 = rvs_sphere_y(θmax; dtype=geometry.dtype)
        # TODO: this inverse transform is wrong
        @. p0 = atan_invtransform(p0, θmax)

        opt = optimize(p -> loss(p, prev_init_direction, θmax), p0, alg, optim_options)
        if isapprox(opt.minimum, 0.0, atol=loss_atol)
            # Transform back to the default coordinate system
            @. opt.minimizer = atan_transform(opt.minimizer, θmax)
            rotate_from_y!(opt.minimizer, prev_init_direction)
            push!(opt.minimizer, geometry.arrival_time, geometry.redshift)

            return opt.minimizer
        else
            θmax0 *= 1.25
        end
    end
    return nothing
end