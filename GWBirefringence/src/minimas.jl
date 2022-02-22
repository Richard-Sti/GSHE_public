import Clustering: kmeans
import Optim: optimize, NLSolversBase.InplaceObjective, only_fg!
import DiffResults: GradientResult

"""
    find_minima(
        floss::Function,
        alg::NelderMead,
        options;
        Nattempts=50
    )
"""
function find_minima(
    floss::Function,
    alg::NelderMead,
    options::Options;
    Nattempts::Int64=50,
    Nsols::Int64=3
)
    X = [zeros(GWFloat, 2)]
    for i in 1:Nsols
        # Optionally pass previously found solutions into the loss func.
        loss(p) = floss(p, (i ===1 ? nothing : X))
        
        Xnew = find_minimum(loss, alg, options; Nmax=Nattempts)
        # Terminate the search
        if Xnew === nothing
            @info ("Search terminated with $(i-1)/$Nsols solutions after "
                   *"trying $Nattempts attempts to find a new solution.")
            break
        end
        Xnew, Xval = Xnew
        # Append the newly found solution
        i === 1 ? (X[1][:] = Xnew) : push!(X, Xnew)
    end
    # Return and turn this into a matrix
    return  mapreduce(permutedims, vcat, X)
end


"""
    find_minimum(
        floss,
        alg::NelderMead,
        options::Options;
        Nmax::Int64=100,
        atol::Float64=1e-12
    )

Find minimum of a function `floss`.
"""
function find_minimum(
    floss::Function,
    alg::NelderMead,
    options::Options;
    Nmax::Int64=500,
    atol::Float64=1e-12
)
    for i in 1:Nmax
        opt = optimize(floss, uniform_sample_sphere(), alg, options)
        if isapprox(opt.minimum, 0.0, atol=atol)
            return opt.minimizer, opt.minimum
        end
    end

    return nothing
end


# """
#     find_minimum(
#         floss,
#         alg::ConjugateGradient,
#         options::Options;
#         Nmax::Int64=100,
#         atol::Float64=1e-12
#     )
# 
# Find minimum of a function `floss` using gradients."""
# function find_minimum(
#     floss,
#     alg::ConjugateGradient,
#     options::Options;
#     Nmax::Int64=100,
#     atol::Float64=1e-12
# )
#     result = GradientResult(zeros(GWFloat, 3))
#     floss_gradient!(F, G, p) = loss_gradient!(F, G, p, result, floss)
#     for i in 1:Nmax
#         opt = optimize(only_fg!(floss_gradient!),
#                        uniform_sample_sphere(true),
#                        alg,
#                        options)
# 
#         if isapprox(opt.minimum, 0.0, atol=atol)
#             Xnew = opt.minimizer
#             cartesian_to_spherical!(Xnew)
#             return Xnew[2:end], opt.minimum
#         end
#     end
#     return nothing
# #    @assert false "No minimum found in $Nmax attempts."
# end


"""
    search_minima(find_minimum::Function, N::Int64)

Search `N` minima from `find_minimum`, which should take no arguments. The
search is parallelised.
"""
function search_minima(find_minimum::Function, N::Int64) 
    X = zeros((N, 3))
    
    Threads.@threads for i in 1:N
        opt = find_minimum()
        X[i, 1:2] = opt.minimizer
        X[i, 3] = opt.minimum
    end
    return X
end


"""
    cluster_minima(
        X::Union{Matrix{GWFloat}, Vector{Vector{GWFloat}}},
        tol::Float64=1e-10
    )

Assign vectors in X of shape (Nsamples, ndim) to clusters using k-means.
Searches for clusters until the cost function is below tolerance and returns
numbers of clusters and assignemnts
"""
function cluster_minima(
    X::Union{Matrix{GWFloat}, Vector{Vector{GWFloat}}},
    tol::Float64=1e-10
)
    if typeof(X) == Vector{Vector{GWFloat}}
        X = mapreduce(permutedims, vcat, X)
    end

    X = mapslices(spherical_to_cartesian, X, dims=2)
    for k in 1:size(X)[1]
        R = kmeans(transpose(X), k; maxiter=500)
        @assert R.converged "Kmeans not converged."
        sum(R.totalcost) < tol ? (return k, R.assignments) : nothing
    end
end


"""
    timing_minima(geo_solver::Function, X::Matrix{GWFloat};
                  atol::Float64=1e-10)

Average solutions from geodesics.
"""
function timing_minima(geo_solver::Function, X::Matrix{GWFloat};
                       atol::Float64=1e-10)
    N, assign = cluster_minima(X[:, 1:2], atol)
    out = zeros(N, 3)
    # Grab the time from the solver
    f(p) = geo_solver(p)[1, end]
    # Mean and std
    mu(x::Vector{GWFloat}) = sum(x) / length(x)
    std(x::Vector{GWFloat}) = sqrt(sum((x .- mu(x)).^2) / length(x))

    for k in 1:N
        times = mapslices(f, X[assign .== k, 1:2], dims=2)
        Nsol = length(times)
        times = reshape(times, Nsol)
        if Nsol > 1
            dsp = std(times)
            @assert dsp < atol "Tolerance not reached. Reached $dsp."
        end
        # Output averaged timing and initial direction
        out[k, 1] = mu(times)
        out[k, 2:3] = mapslices(mu, X[assign .== k, 1:2], dims=1)
    end
    return out
end


"""
    match_Xminus(Xplus::Matrix{GWFloat}, Xminus::Matrix{GWFloat})

Match the `Xminus` matrix of minimas to `Xplus`. Assumes that these
matrices originate from `GWBirefringence.average_clusters`.
"""
function match_Xminus(Xplus::Matrix{GWFloat}, Xminus::Matrix{GWFloat})
    Nplus = size(Xplus)[1]
    Nminus = size(Xminus)[1]
    @assert Nplus == Nminus  "Different number of minimas, $Nplus and $Nminus."

    out = zero(Xminus)
    for i in 1:Nplus
        if i == Nplus
            out[i, :] = Xminus
            break
        end
        # Calculate Cartesian vectors
        X = vcat(Xplus[i:i, 2:3], Xminus[:, 2:3])
        X = mapslices(spherical_to_cartesian, X, dims=2)

        R = kmeans(transpose(X), 2; maxiter=500)
        assign = R.assignments
        # Find which minima matches
        mask = (assign .== assign[1])[2:end]
        @assert sum(mask) == 1 "Matched multiple minimas"
        out[i, :] = Xminus[mask, :]
        Xminus = Xminus[.~mask, :]
    end
    return out
end


"""
    calc_flip(step::Int64, s_index::Int64, X::Array{GWFloat, 4})

Calculates whether to flip the solution at `step + 1` with respect to the
solution at `step` for polarisation indexed by `s_index`.

Determines whether to do the flip on the angular distance between the two
initial directions.
"""
function calc_flip(step::Int64, s_index::Int64, X::Array{GWFloat, 4})
    x0 = X[step, s_index, 1, 2:end]
    auto = angdist(x0, X[step+1, s_index, 1, 2:end])
    cross = angdist(x0, X[step+1, s_index, 2, 2:end])
    return auto > cross
end


"""
    do_flip!(
        Zscratch::Matrix{GWFloat},
        X::Array{GWFloat, 4},
        index::Int64
    )

Flips to which geodesics the solutions belong.
"""
function do_flip!(
    Zscratch::Matrix{GWFloat},
    X::Array{GWFloat, 4},
    step::Int64
)
    Zscratch[:, :] = X[step, :, 1, :]
    X[step, :, 1, :] = X[step, :, 2, :]
    X[step, :, 2, :] = Zscratch[:, :]
end


"""
    do_flips!(X::Array{GWFloat, 4})

Flips to which gedeosics solutions belong in the array `X`.
"""
function do_flips!(X::Array{GWFloat, 4})
    N = size(X)[1]
    Zscratch = zeros(GWFloat, 2, 3)

    for step in 1:(N-1)
        f1 = calc_flip(step, 1, X)
        f2 = calc_flip(step, 2, X)
        @assert (f1 == f2) "Different flips for the two polarisations."

        f1 ? do_flip!(Zscratch, X, step+1) : nothing
    end
end

"""
    calc_dts(X::Array{GWFloat, 4})

Calculates the time delays from `X`.
"""
function calc_dts(X::Array{GWFloat, 4})
    return abs.(X[:, 1, :, 1] - X[:, 2, :, 1])
end
