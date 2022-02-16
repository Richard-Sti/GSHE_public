import Clustering: kmeans
import Optim: optimize, NelderMead, Options, NLSolversBase.InplaceObjective, ConjugateGradient, only_fg!


"""
    find_minimum(
        floss,
        alg::NelderMead,
        options::Options;
        Nmax::Int64=100,
        atol::AbstractFloat=1e-12
    )

Find minimum of a function `floss`.
"""
function find_minimum(
    floss,
    alg::NelderMead,
    options::Options;
    Nmax::Int64=100,
    atol::AbstractFloat=1e-12
)
    for i in 1:Nmax
        opt = optimize(floss,
                       uniform_sample_sphere(),
                       alg,
                       options)

        if isapprox(opt.minimum, 0.0, atol=atol)
            return opt
        end
    end
    @assert false "No minimum found in $Nmax attempts."
end


"""
    find_minimum(
        floss,
        alg::Union{NeldearMead, ConjugateGradient},
        options::Options;
        Nmax::Int64=100,
        atol::AbstractFloat=1e-12
    )

Find minimum of a function `floss` using gradients."""
function find_minimum(
    floss,
    alg::ConjugateGradient,
    options::Options;
    Nmax::Int64=10,
    atol::AbstractFloat=1e-12
)
    for i in 1:Nmax
        opt = optimize(only_fg!(floss),
                       uniform_sample_sphere(true),
                       alg,
                       options)

        if isapprox(opt.minimum, 0.0, atol=atol)
            return opt
        end
    end
    @assert false "No minimum found in $Nmax attempts."
end


"""
    search_minima(find_minimum::Function, N::Int64)

Search `N` minima from `find_minimum`, which should take no arguments. The
search is parallelised.
"""
function search_minima(find_minimum::Function, N::Int64) 
    X = zeros((N, 4))
    
    Threads.@threads for i in 1:N
        opt = find_minimum()
        X[i, 1:3] = opt.minimizer
        X[i, 4] = opt.minimum
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
    search_unique_minima(fmin::Function, N::Int64;
                         Nmax::Int64=50, atol::AbstractFloat=1e-12)

Searches ``N`` unique minimas.
"""
function search_unique_minima(fmin::Function, N::Int64;
                              Nmax::Int64=50, atol::AbstractFloat=1e-12)
    out = zeros(N, 2)
    Nfound = 0
    X = [zeros(GWFloat, 2)]
    for i in 1:Nmax
        opt = fmin()
        dir = opt.minimizer
        push!(X, dir)
        i == 1 ? (X[1][:]= dir) : push!(X, dir)
        
        k, __ = cluster_minima(X, atol)
        if k > Nfound
            Nfound += 1
            out[Nfound, :] = dir 
            
            Nfound == N ? (return out) : nothing
        end
    end
    @assert false "Failed to find $N minima in $Nmax attempts. Found $Nfound."
end


"""
    timing_minima(geo_solver::Function, X::Matrix{GWFloat};
                  atol::AbstractFloat=1e-10)

Average solutions from geodesics.
"""
function timing_minima(geo_solver::Function, X::Matrix{GWFloat};
                       atol::AbstractFloat=1e-10)
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
"""
function do_flip!(
    Zscratch::Matrix{GWFloat},
    X::Array{GWFloat, 4},
    index::Int64
)
    Zscratch[:, :] = X[index, :, 1, :]
    X[index, :, 1, :] = X[index, :, 2, :]
    X[index, :, 2, :] = Zscratch[:, :]
end


"""
    do_flips!(X::Array{GWFloat, 4})
"""
function do_flips!(X::Array{GWFloat, 4})
    N = size(X)[1]
    Zscratch = zeros(GWFloat, 2, 3)

    for step in 1:(N-1)
        f1 = calc_flip(step, 1, X)
        f2 = calc_flip(step, 2, X)
        @assert f1 == f2
        if f1 == true
            do_flip!(Zscratch, X, step+1)
        end
    end
end
