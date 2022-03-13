import Clustering: kmeans
import Optim: optimize, NLSolversBase.InplaceObjective, only_fg!
import DiffResults: GradientResult


"""
    find_minima(
        geometry::Geometry,
        alg::NelderMead,
        options::Optim.Options;
        Nsols::Int64=1,
        Nattempts=50
    )

Find `Nsols` minima of a geodesic.
"""
function find_minima(
    geometry::Geometry,
    alg::NelderMead,
    options::Options;
    Nsols::Int64=1,
    Nattempts::Int64=100,
)
    X = nothing
    floss = setup_geodesic_loss(geometry)
    f(p::Vector{GWFloat}) = floss(p, X)
    for i in 1:Nsols
        # Optionally pass previously found solutions into the loss func.
        Xnew = find_minimum(f, geometry, alg, options; Nmax=Nattempts)
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
    find_minimum(
        floss::Function,
        geometry::Geometry,
        alg::NelderMead,
        options::Optim.Options;
        Nmax::Int64=100,
        atol::Float64=1e-12
    )

Find minimum of a geodesic function `floss(p::Vector{GWFloat})`.
"""
function find_minimum(
    floss::Function,
    geometry::Geometry,
    alg::NelderMead,
    options::Options;
    Nmax::Int64=500,
    atol::Float64=1e-12
)
    for i in 1:Nmax
        opt = optimize(floss, rvs_sphere(), alg, options)
        if isapprox(opt.minimum, 0.0, atol=atol)
#            return opt.minimizer
            x = opt.minimizer
            push!(x, geometry.xf[1])
            return x
        end
    end

    return nothing
end


"""
    θmax_scaling(θmax0::GWFloat, ϵ::GWFloat)

Calculate :math:`θmax = θmax0 + √ϵ`, however maximum value is capped at π/3.
"""
function θmax_scaling(θmax0::GWFloat, ϵ::GWFloat)
    θmax =  θmax0 + 0.75 * sqrt(ϵ)
    θmax > π / 3 ? (return π/3) : return θmax
end


"""
    find_restricted_minimum(
        geometry::Geometry,
        pfound::Vector{GWFloat},
        alg::NelderMead,
        options::Options;
        θmax0::GWFloat=0.025,
        Nmax::Int64=500,
        atol::Float64=1e-12
    )

Find a spin Hall minimum. Searches withing `θmax` angular distance of `pfound`.
"""
function find_restricted_minimum(
    geometry::Geometry,
    pfound::Vector{GWFloat},
    alg::NelderMead,
    options::Options;
    θmax0::GWFloat=0.025,
    Nmax::Int64=500,
    atol::Float64=1e-12
)
    loss = GWBirefringence.setup_spinhall_loss(geometry)
    for i in 1:Nmax
        θmax = θmax_scaling(θmax0, geometry.params.ϵ)
        # Sample initial position and inv transform it
        p0 = rvs_sphere_y(θmax)
        @. p0 = atan_invtransform(p0, θmax)
        f(p::Vector{GWFloat}) = loss(p, pfound, θmax)

        opt = optimize(f, p0, alg, options)
        if isapprox(opt.minimum, 0.0, atol=atol)
            # Transform back to the default coordinate system
            x = opt.minimizer
            @. x = atan_transform(x, θmax)
            rotate_from_y!(x, pfound)
            push!(x, geometry.xf[1])
            return x
        else
            θmax0 *= 1.25
        end
    end

    return nothing
end