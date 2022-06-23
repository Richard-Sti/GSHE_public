"""
    predict_llsq(x::Real, pars::Vector{<:Real})

Predict a linear least squares fit at x.
"""
function predict_llsq(x::Real, pars::Vector{<:Real})
    return pars[2] + pars[1] * x
end


function predict_llsq(x::Vector{<:Real}, pars::Vector{<:Real})
    return pars[2] .+ pars[1] .* x
end


"""
    residuals_llsq(x::Real, y::Real, pars::Vector{<:Real})

Residuals of a linear least squares fit at x.
"""
function residuals_llsq(x::Real, y::Real, pars::Vector{<:Real})
    return predict_llsq(x, pars) - y
end


function residuals_llsq(x::Vector{<:Real}, y::Vector{<:Real}, pars::Vector{<:Real})
    return predict_llsq(x, pars) - y
end


"""
    R2_llsq(x::Vector{<:Real}, y::Vector{<:Real}, pars)

R^2 score of a linear least squares fit.
"""
function R2_llsq(x::Vector{<:Real}, y::Vector{<:Real}, pars::Vector{<:Real})
    return 1 - sum(residuals_llsq(x, y, pars).^2) / sum((y .- mean(y)).^2)
end


"""
    findoutlier_fixedslope(x::Vector{<:Real}, y::Vector{<:Real}, slope::Real=2)

Find outliers where it is assumed that y = C + slope * x with known slope. Clusters the
inferred C into 2 bins and sets the outlier bins to be the one whose slope is further away
from the assumed slope. Returns the index of the worst MAE point.
"""
function findoutlier_fixedslope(x::Vector{<:Real}, y::Vector{<:Real}, slope::Real=2)
    # Begin by assigning to clusters via K-means
    N = length(x)
    X = reshape(y .- slope * x, 1, N)
    R = Clustering.kmeans(X, 2; maxiter=200)
    @assert Clustering.nclusters(R) == 2
    a = Clustering.assignments(R)

    # Guess that the cluster with fewer points is outliers
    cnts = Clustering.counts(R)
    cnts[1] < cnts[2] ? (isout = a .== 1) : (isout = a .== 2)

    # If 1 outlier directly return it
    if sum(isout) == 1
        return (1:N)[isout][1]
    end

    outpars = llsq(x[isout], y[isout])
    inpars = llsq(x[.~isout], y[.~isout])
    # Check if inliers slope is closer to the expected value. If not flip
    if abs(outpars[1] - slope) < abs(inpars[1] - slope)
        @. isout = ~isout
        inpars = outpars
    end

    # Residuals of the outliers w.r.t. to the inliers fit
    res = residuals_llsq(x[isout], y[isout], inpars)
    __ , worstind = findmax(abs.(res))
    # Index of the worst outlier
    return (1:N)[isout][worstind]
end


"""
    move_point!(x0, xf, p)

Move point `p` from vector `x0` to vector `xf`.
"""
function move_point!(x0, xf, p)
    @assert p in x0 "`p` must be in the initial vector `x0`."
    deleteat!(x0, findall(x -> x == p, x0))
    push!(xf, p)
end


"""
    find_outliers(
        x::Vector{<:Real},
        y::Vector{<:Real},
        R2tol::Real,
        minpoints::Integer,
        slope::Real
    )

Calculate outliers from points that are assumed to have been generated as y = m + slope * x,
where the slope is assumed to be approximately known. Clusters points into outliers and
sets a point as outlier if its removal sufficiently reduces the R2 score.
"""
function find_outliers(
    x::Vector{<:Real},
    y::Vector{<:Real},
    R2tol::Real,
    minpoints::Integer,
    slope::Real
)
    N = length(x)
    outliers = [i for i in 1:N if isnan(y[i])]
    inliers = [i for i in 1:N if ~isnan(y[i])]

    # Consider everything as outliers if less than minpoints
    if length(inliers) < minpoints
        push!(outliers, inliers...)
        return outliers
    end

    # At least 3 unique points for a fit
    while length(inliers) > 2
        # Calculate old R2 and worst outlier
        oldR2 = R2_llsq(x[inliers], y[inliers], llsq(x[inliers], y[inliers]))
        worstind = inliers[findoutlier_fixedslope(x[inliers], y[inliers], slope)]

        # Move the worst point to outliers and calculate the ΔR2
        move_point!(inliers, outliers, worstind)
        ΔR2 = R2_llsq(x[inliers], y[inliers], llsq(x[inliers], y[inliers])) - oldR2

        # If ΔR2 less than tolerance move point back to inliers and finish
        if ΔR2 < R2tol
            move_point!(outliers, inliers, worstind)
            break
        end
    end


    # If only 2 inliers left set them as outliers
    if length(inliers) == 2
        for inlier in inliers
            push!(outliers, inlier)
        end
    end

    sort!(outliers)
    return outliers
end


"""
    find_outliers(
        Xgshe::Matrix{<:Real},
        Xgeo::Vector{<:Real},
        geometry::Geometry,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        increasing_ϵ::Bool
    )

Find outliers in the GSHE to geodesic timing. Moreover anything flagged as NaN is also
considered an outlier. Checks whether the inliers follow an expected slope specified in
geometry.
"""
function find_outliers(
    Xgshe::Matrix{<:Real},
    Xgeo::Vector{<:Real},
    geometry::Geometry,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    increasing_ϵ::Bool
)
    @unpack integration_error, R2tol, minpoints, expslope = geometry.postproc_options
    N = length(ϵs)
    logϵs = log10.(ϵs)
    Δts = abs.(Xgshe[:, 3] .- Xgeo[3])

    # Set anything below the integration error to NaN
    mask = cut_below_integration_error(logϵs, Δts, integration_error; mask_only=true, verbose=false)
    Δts[.~mask] .= NaN

    # Outliers and inliers (not NaN and not in outliers)
    outliers = find_outliers(logϵs, log10.(Δts), R2tol, minpoints, expslope)
    if ~increasing_ϵ && N in outliers
        deleteat!(outliers, findall(x -> x == N, outliers))
    end

    inliers = [i for i in 1:N if (~isnan(Δts[i]) && ~(i in outliers))]
    return outliers, inliers
end


"""
    check_solutions!(
        Xgshe::Matrix{<:Real},
        Xgeo::Vector{<:Real},
        s::Integer,
        geometry::Geometry,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        increasing_ϵ::Bool
    )

Check whether any outliers and if so attempt to recalculate them.
"""
function check_solutions!(
    Xgshe::Matrix{<:Real},
    Xgeo::Vector{<:Real},
    s::Integer,
    geometry::Geometry,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    increasing_ϵ::Bool
)
    @assert is_strictly_increasing(ϵs) "ϵ must be strictly increasing."; flush(stdout)
    @unpack Ncorr, verbose = geometry.postproc_options
    nloops = Xgeo[6]

    for k in 1:(Ncorr + 1)
        # If decreasing search and geodesic is NaN repeat everything
        if ~increasing_ϵ && any(isnan.(Xgeo))
            newXgeo, newXgshe = solve_decreasing(Xgshe[end, :], geometry, s, ϵs, verbose=false, force_nocheck=true)
            Xgeo .= newXgeo
            Xgshe .= newXgshe
            continue
        end

        # If no outliers exit the correcting attempts
        outliers, inliers = find_outliers(Xgshe, Xgeo, geometry, ϵs, increasing_ϵ)
        length(outliers) == 0 && break

        # Exit if too many attempts
        if k == Ncorr + 1
            verbose && @warn "Failed to correct $(length(outliers)) outliers."; flush(stdout)
            break
        end

        # If decreasing reverse outliers
        ~increasing_ϵ && reverse!(outliers)

        for outlier in outliers
            # Set the iterators and limiting guess
            if increasing_ϵ
                # Search from below the outlier
                iters = reverse(1:outlier - 1)
                dir = Xgeo[1:2]
                ϵ0 = 0
                nrepeats = Xgeo[8]
            else
                # Search from above the outlier
                iters = outlier + 1:length(ϵs) - 1
                dir = Xgshe[end, 1:2]
                ϵ0 = ϵs[end]
                nrepeats = Xgshe[end, 8]
            end


            for jj in iters
                if jj in inliers
                    dir .= Xgshe[jj, 1:2]
                    ϵ0 = ϵs[jj]
                    nrepeats = Xgshe[jj, 8]
                    break
                end
            end
            Xgshe[outlier, :] .= find_consecutive_minimum(geometry, ϵs[outlier], s, dir, ϵ0, nloops, nrepeats)

            # Move the newly calculated outlier to inliers
            if ~any(isnan.(Xgshe[outlier, 1:2]))
                push!(inliers, outlier)
            end
        end
    end

    return nothing
end


"""
    check_geodesics(Xgeo::Matrix{<:Real}, geometry::Geometry)

Check that the geodesics of the decreasing search are in agreement.
"""
function check_geodesics(Xgeo::Matrix{<:Real}, geometry::Geometry)
    @unpack geodesics_Δσ, geodesics_Δt, check_verbose = geometry.postproc_options

    Δσ = GSHEIntegrator.angdist(Xgeo[1, 1:2], Xgeo[2, 1:2])
    Δt = abs(Xgeo[1, 3] - Xgeo[2, 3])

    if Δσ < geodesics_Δσ && Δt < geodesics_Δt
        return Xgeo[1, :]
    else
        if check_verbose
            println("Geodesics from downwards ϵ ladder did not converge, Δσ = $Δσ, Δt = $Δt"); flush(stdout)
        end
        return fill(NaN, size(Xgeo, 2))
    end
end
