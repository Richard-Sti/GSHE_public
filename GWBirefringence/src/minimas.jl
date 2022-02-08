import Clustering: kmeans
import Optim: optimize, NelderMead


"""
    search_minima(N::Int64, fmin::Function, options,
                       algorithm=NelderMead(), atol::Float64=1e-12)

Search `N` minima of the `fmin`.
"""
function search_minima(N::Int64, fmin::Function, options,
                       algorithm=NelderMead(), atol::Float64=1e-12)
    X = zeros((N, 3))
    
    Threads.@threads for i in 1:N
        while true
            p0 = GWBirefringence.uniform_sample_sphere()
            opt = optimize(fmin, p0, algorithm, options)
            if isapprox(opt.minimum, 0.0, atol=1e-12)
                X[i, 1:2] = opt.minimizer
                X[i, 3] = opt.minimum
                break
            end
        end
    end
    
    return X
end


"""
    assign_clusters(X::Matrix, tol::Float64=1e-10)

Assign vectors in X of shape (Nsamples, ndim) to clusters using k-means.
Searches for clusters until the cost function is below tolerance and returns
cluster assignemnts.
"""
function assign_clusters(X::Matrix{GWFloat}, tol::Float64=1e-10)
    Xcart = mapslices(spherical_to_cartesian, X, dims=(2))

    for k in 1:size(X)[1]
        R = kmeans(transpose(Xcart), k; maxiter=500)
        @assert R.converged "Kmeans convergence"
        
        if sum(R.costs) < tol
            return R.assignments
        end
    end
end


"""
    average_clusters(geo_solver::Function, X::Matrix{GWFloat},
                     tol::Float64=1e-10)

Average solutions from many geodesics.
"""
function average_clusters(geo_solver::Function, X::Matrix{GWFloat},
                          tol::Float64=1e-10)
    assign = assign_clusters(X[:, 1:2])

    N = length(unique(assign))
    out = zeros(N, 3)

    # Grab the time from the solver
    f(p) = geo_solver(p)[1, end]
    # Mean and std
    mu(x::Vector) = sum(x) / length(x)
    std(x::Vector) = sqrt(sum((x .- mu(x)).^2) / length(x))

    for k in 1:N
        times = mapslices(f, X[assign .== k, 1:2], dims=(2))
        times = reshape(times, (length(times)))
        @assert std(times) < tol "Tolerance not reached"
        out[k, 1] = mu(times)
        out[k, 2:3] = mapslices(mu, X[assign .== k, 1:2], dims=(1))
    end
    out
end


"""
    match_Xminus(Xplus::Matrix{GWFloat}, Xminus::Matrix{GWFloat})

Match the `Xminus` matrix of minimas to `Xplus`. Assumes that these
matrices originate from `GWBirefringence.average_clusters`.
"""
function match_Xminus(Xplus::Matrix{GWFloat}, Xminus::Matrix{GWFloat})
    Xminus = copy(Xminus)
    N = size(Xplus)[1]
    @assert N == size(Xminus)[1] "Different number of minimas"
    out = zero(Xminus)

    for i in 1:N
        if i == N
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
