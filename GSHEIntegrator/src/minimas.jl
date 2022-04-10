"""
    find_minima(
        geometry::Geometry,
        alg::NelderMead,
        options::Optim.Options;
        Nsols::Integer=1,
        Nattempts::Integer=50
    )

Find `Nsols` minima of a geodesic for a given geometry.
"""
function find_geodesic_minima(
    geometry::Geometry,
    alg::NelderMead,
    options::Options;
    Nsols::Integer=1,
    Nattempts::Integer=100,
)
    X = nothing
    floss = setup_geodesic_loss(geometry)
    f(p::Vector{<:Real}) = floss(p, X)
    for i in 1:Nsols
        # Optionally pass previously found solutions into the loss func.
        Xnew = find_unconstrained_minimum(f, geometry, alg, options; Nmax=Nattempts)
        # Terminate the search
        if Xnew === nothing
            @info ("Search terminated with $(i-1)/$Nsols solutions after "
                   *"trying $Nattempts attempts to find a new solution.")
            break
        end
        # Append the newly found solution
        if i === 1
            X = [Xnew]
        else
            push!(X, Xnew)
        end
    end
    # Return and turn this into a matrix
    return  mapreduce(permutedims, vcat, X)
end


"""
    find_unconstrained_minimum(
        floss::Function,
        geometry::Geometry,
        alg::NelderMead,
        options::Optim.Options;
        Nmax::Integer=100,
        atol::Real=1e-12
    )

Find minimum of a loss function ``floss`` whose argument is a length 2 vector of spherical
angles.
"""
function find_unconstrained_minimum(
    floss::Function,
    geometry::Geometry,
    alg::NelderMead,
    options::Options;
    Nmax::Integer=500,
    atol::Real=1e-12
)
    for i in 1:Nmax
        opt = optimize(floss, rvs_sphere(type=geometry.type), alg, options)
        if isapprox(opt.minimum, 0.0, atol=atol)
            x = opt.minimizer
            push!(x, geometry.arrival_time, geometry.redshift)
            return x
        end
    end

    return nothing
end


"""
    θmax_scaling(θmax0::Real, ϵ::Real)

Calculate :math:`θmax = θmax0 + √ϵ`, however maximum value is capped at π/3.
"""
function θmax_scaling(θmax0::Real, ϵ::Real)
    θmax =  θmax0 + 0.75 * sqrt(ϵ)
    θmax > π / 3 ? (return π/3) : return θmax
end


"""
    find_restricted_minimum(
        geometry::Geometry,
        pfound::Vector{<:Real},
        alg::NelderMead,
        options::Options;
        θmax0::Real=0.025,
        Nmax::Integer=500,
        atol::Real=1e-12
    )

Find a spin Hall minimum. Searches withing `θmax` angular distance of `pfound`.
"""
function find_restricted_minimum(
    geometry::Geometry,
    pfound::Vector{<:Real},
    alg::NelderMead,
    options::Options;
    θmax0::Real=0.025,
    Nmax::Integer=500,
    atol::Real=1e-12
)
    loss = setup_spinhall_loss(geometry)
    for i in 1:Nmax
        θmax = θmax_scaling(θmax0, geometry.params.ϵ)
        # Sample initial position and inv transform it
        p0 = rvs_sphere_y(θmax; type=geometry.type)
        @. p0 = atan_invtransform(p0, θmax)
        f(p::Vector{<:Real}) = loss(p, pfound, θmax)

        opt = optimize(f, p0, alg, options)
        if isapprox(opt.minimum, 0.0, atol=atol)
            # Transform back to the default coordinate system
            x = opt.minimizer
            @. x = atan_transform(x, θmax)
            rotate_from_y!(x, pfound)
            push!(x, geometry.arrival_time, geometry.redshift)
            return x
        else
            θmax0 *= 1.25
        end
    end

    return nothing
end