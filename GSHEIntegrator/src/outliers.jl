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
    ypred = predict_llsq(x, pars)
    return 1 - sum((ypred .- y).^2) / sum((ypred .- GSHEIntegrator.average(y)).^2)
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
    find_outliers_llsq(x::Vector{<:Real}, y::Vector{<:Real}, R2tol::Real)

Find outliers in a linear least squares fit outliers are identified as points whose removal
reduces the R^2 score by more than `R2tol`.
"""
function find_outliers_llsq(x::Vector{<:Real}, y::Vector{<:Real}, R2tol::Real)
    N = length(x)
    outliers = Vector{Int64}()
    inliers = [i for i in 1:N if ~isnan(y[i])]

    # We need at least two unique points for the linear fit
    while length(inliers) > 2
        # Fit the points and calculate the old R2 prior to removing the worst point
        oldpars = llsq(x[inliers], y[inliers])
        oldR2 = R2_llsq(x[inliers], y[inliers], oldpars)

        # Find the worst point along with its index in inliers
        __, maxind = findmax(abs.(residuals_llsq(x[inliers], y[inliers], oldpars)))
        maxind = inliers[maxind]

        # Move the worst point to from inliers to outliers
        move_point!(inliers, outliers, maxind)

        # Calculate the new R2
        newpars = llsq(x[inliers], y[inliers])
        newR2 = R2_llsq(x[inliers], y[inliers], newpars)

        # If the average decreased be less than some tolerance move the point back to
        # inliers and terminate the loop.
        ΔR2 = newR2 - oldR2
        if ΔR2 < R2tol
            move_point!(outliers, inliers, maxind)
            break
        end
    end

    # In case only 2 inliers left automatically consider these as outliers
    if length(inliers) == 2
        for inlier in inliers
            push!(outliers, inlier)
        end
    end

    return outliers
end


"""
    check_gshes!(
        Xgshe::Array{<:Real, 3},
        Xgeo::Vector{<:Real},
        geometry::Geometry,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}}
    )

Checks for outliers in the GSHE to geodesic time delay, if any attempts to recalculate them.
"""
function check_gshes!(
    Xgshe::Array{<:Real, 3},
    Xgeo::Vector{<:Real},
    geometry::Geometry,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}}
)
    log_ϵs = log10.(ϵs)
    @assert is_strictly_increasing(log_ϵs) "ϵ must be strictly increasing."
    flush(stdout)
    N = length(log_ϵs)

    @unpack integration_error, R2tol, Ncorrect = geometry.postproc_options
    @unpack gshe_convergence_verbose = geometry.opt_options
    for s in 1:2
        for k in 1:(Ncorrect + 1)
            Δts = abs.(Xgshe[s, :, 3] .- Xgeo[3])
            # TODO don't need log_ϵs here
            mask = cut_below_integration_error(log_ϵs, Δts; mask_only=true, verbose=false)
            # Set anything below the integration error to NaN
            Δts[.~mask] .= NaN

            # Calculate the outliers
            outliers = find_outliers_llsq(log_ϵs, log10.(Δts), R2tol)
            # Get the GSHE sols that are not NaNs and not outliers
            good_gshes = [i for i in 1:N if (~isnan(Δts[i]) && ~(i in outliers))]

            # If no outliers exit the correcting attempts
            if length(outliers) == 0
                break
            end

            # Exit if too many attempts
            if k == Ncorrect + 1
                if gshe_convergence_verbose
                    @warn ("Failed to recalculate outliers $outliers for s=$s. "
                           *"Setting to NaN, either inspect the solutions or increase `R2tol`.")
                    flush(stdout)
                end

                # Set the values it failed to recalculate to NaNs
                for j in outliers
                    Xgshe[s, j, :] .= NaN
                end
                # Exit
                break
            end

            # Ensure outliers are sorted
            sort!(outliers)
            for outlier in outliers
                # Get the previous good solution. Begin by proposing the geodesic direction
                p0 = Xgeo[1:2]
                for jj in reverse(1:outlier)
                    # If reversed to 1 do not execute the code below
                    if jj == 1
                        break
                    end

                    # If the previous point is not a good one continue
                    if ~(jj - 1 in good_gshes)
                        continue
                    end
                    # Propose the previous point as initial direction
                    p0 = Xgshe[s, jj - 1, 1:2]
                end

                Xgshe[s, outlier, :] .= find_restricted_minimum(
                    geometry, ϵs[outlier], s == 1 ? geometry.s : -geometry.s, p0)
            end
        end
    end

    return nothing
end
