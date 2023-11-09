"""
    solve_initial(geometry::Geometry{<:Real}, ϵ::Real, Nsols::Integer=2)

Find the initial (unrestricted) initial shooting direction either for GSHE or geodesics,
depending on ϵ for a single geometry. If ϵ = 0 the output shape is
(Nsols, 6), otherwise (Nsols, 2, 6).
"""
function solve_initial(geometry::Geometry{<:Real}, ϵ::Real, Nsols::Integer=2)
    Xright = find_initial_minima(geometry, ϵ, geometry.s, Nsols)

    if ϵ == 0
        return Xright
    end

    @assert Nsols <= 2 "At most `Nsols` of 2 supported for ϵ != 0."
    Xleft = find_initial_minima(geometry, ϵ, -geometry.s, Nsols)

    # Check the ordering of the solutions from the minimisers
    # We expect that for a given geodesic the GSHE initial conditions and arrival are similar
    if Nsols == 2
        # Function that gives 2 if i = 1 and 2 if i = 1
        f(i::Integer) = i == 1 ? 2 : 1
        Δt = mean([abs(Xright[i, 3] - Xleft[i, 3]) for i in 1:2])
        Δtflip = mean([abs(Xright[i, 3] - Xleft[f(i), 3]) for i in 1:2])
        Δσ = mean([angdist(Xright[i, 1:2], Xleft[i, 1:2]) for i in 1:2])
        Δσflip = mean([angdist(Xright[i, 1:2], Xleft[f(i), 1:2]) for i in 1:2])

        toflip = Δσ > Δσflip

        if toflip != (Δt > Δtflip)
            @warn "Δt = $Δt and Δσ = $Δσ do not agree"; flush(stdout)
        end

        if toflip
            reverse!(Xleft, dims=1)
        end
    end

    # Carefully put into an array
    X = zeros(geometry.dtype, Nsols, 2, size(Xright, 2))
    for i in 1:Nsols
        X[i, 1, ..] .= Xright[i, ..]
        X[i, 2, ..] .= Xleft[i, ..]
    end

    return X
end


"""
    solve_initial(
        geometries::Vector{<:Geometry{<:Real}},
        ϵ::Real,
        Nsols::Integer=2,
        verbose::Bool=true,
    )

Find the initial (unrestricted) initial shooting direction either for GSHE or geodesics,
depending on ϵ for the specified geometries. If ϵ = 0 the output shape is
(Nconfigurations, Nsols, 6), otherwise (Nconfigurations, Nsols, 2, 6).
"""
function solve_initial(
    geometries::Vector{<:Geometry{<:Real}},
    ϵ::Real,
    Nsols::Integer=2,
    verbose::Bool=true,
)
    check_geometry_dtypes(geometries)
    dtype = geometries[1].dtype
    N = length(geometries)
    X = Vector{Array{dtype}}(undef, N)

    verbose && ϵ == 0 ? (kind = "geodesic") : (kind = "GSHE")
    verbose ? (println("Solving $N configurations' initial $kind directions."); flush(stdout)) : nothing

    Threads.@threads for i in shuffled_iterators(N)
        verbose ? (println("Solving initial $kind direction $i/$N"); flush(stdout)) : nothing
        X[i] = solve_initial(geometries[i], ϵ, Nsols)
    end

    return toarray(X)
end


"""
    is_strictly_increasing(x::Union{Vector{<:Real}, LinRange{<:Real}})

Check if a vector is strictly increasing.
"""
function is_strictly_increasing(x::Union{Vector{<:Real}, LinRange{<:Real}})
    return all((x[i+1] - x[i]) > 0 for i in 1:length(x)-1)
end


"""
function solve_decreasing(
    Xmax::Vector{<:Real},
    geometry::Geometry{<:Real},
    s::Integer,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
    force_nocheck::Bool=false
)

Solve GSHE trajectories for a given geodesic and polarisation, starting from the highest ϵ
which has been calculated beforehand. Returns the geodesic results of shape (6, ) and the
GSHE results of shape (Nϵs, 6).
"""
function solve_decreasing(
    Xmax::Vector{<:Real},
    geometry::Geometry{<:Real},
    s::Integer,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
    force_nocheck::Bool=false
)
    @assert is_strictly_increasing(ϵs) "`ϵs` must be strictly increasing."
    Nϵs = length(ϵs)
    X = fill!(zeros(geometry.dtype, Nϵs, length(Xmax)), NaN)
    # Assign to the highest ϵ the Xmax value
    X[Nϵs, :] .= Xmax
    nloops = Xmax[6]

    # We loop over the ϵs in reverse
    for (i, ϵ) in enumerate(reverse(ϵs))
        verbose && println("$(round(i / Nϵs * 100, digits=2))%, ϵ=$(round(ϵ, sigdigits=2, base=10))"); flush(stdout)

        # Make sure the looping index points to the right places (since we reverse ϵs)
        # So that i = Nϵs, Nϵs - 1, ... , 1
        i = 1 + Nϵs - i
        # Continue to next ϵ if the highest ϵ
        i == Nϵs && continue

        # Find the next high ϵ that has a solution, at most at the next 5 higher sols
        for k in (i + 1):min(i + 1 + 5, Nϵs)
            p0 = X[k, 1:2]
            if ~any(isnan.(p0))
                X[i, :] .= find_consecutive_minimum(geometry, ϵ, s, p0, ϵs[k], nloops, X[k, 8])
                break
            end
        end
    end

    # Now find the geodesic solution. Loop through the min 5 ϵs
    Xgeo = fill!(zeros(geometry.dtype, length(Xmax)), NaN)
    for k in 1:min(5, Nϵs)
        p0 = X[k, 1:2]
        if ~any(isnan.(p0))
            Xgeo[:] .= find_consecutive_minimum(geometry, 0, 2, p0, ϵs[k], nloops, X[k, 8])
            break
        end
    end

    if ~force_nocheck && geometry.postproc_options.check_sols
        check_solutions!(X, Xgeo, s, geometry, ϵs, false)
    end

    return Xgeo, X
end

"""
    _solve_decreasing(
        Xmax::Matrix{<:Real},
        geometry::Geometry{<:Real},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
        verbose::Bool=true,
    )

Solve GSHE trajectories for a geodesic, starting from the highest ϵ which has been
calculated beforehand. Returns the geodesic results of shape (2, 6) and the GSHE results
of shape (2, Nϵs, 6). Should not be used directly as it does not apply any checks and because
it returns two geodesic solutions from each ϵ ladder.
"""
function _solve_decreasing(
    Xmax::Matrix{<:Real},
    geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)
    Xgshes = zeros(geometry.dtype, 2, length(ϵs), size(Xmax, 2))
    Xgeos = zeros(geometry.dtype, 2, size(Xmax, 2))

    for (sx, s) in enumerate([+geometry.s, -geometry.s])
        Xgeo, Xgshe = solve_decreasing(Xmax[sx, :], geometry, s, ϵs; verbose=verbose)
        Xgeos[sx, ..] .= Xgeo
        Xgshes[sx, ..] .= Xgshe
    end

    return Xgeos, Xgshes
end


"""
function solve_decreasing(
    Xmax::Matrix{<:Real},
    geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)

Solve GSHE trajectories for a geodesic, starting from the highest ϵ which has been
calculated beforehand. Returns the geodesic results of shape (5, ) and the GSHE results
of shape (2, Nϵs, 5).
"""
function solve_decreasing(
    Xmax::Matrix{<:Real},
    geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)
    Xgeo, Xgshe = _solve_decreasing(Xmax, geometry, ϵs; verbose=verbose)

    @unpack check_sols, verbose, Ncorr = geometry.postproc_options
    if ~check_sols
        # Live dangerously and just return one geodesic
        return Xgeo[1, :], Xgshe
    end

    # Check that the geodesic ladder converged
    for i in 1:Ncorr
        Xgeo = check_geodesics(Xgeo, geometry)
        # Check if the geodesics agreed. If not attempt to recalculate
        if any(isnan.(Xgeo[.., 3]))
            verbose && println("Geodesics do not agree, recalculating $i/$Ncorr."); flush(stdout)
            Xgeo, Xgshe = _solve_decreasing(Xmax, geometry, ϵs; verbose=verbose)
        else
            break
        end
    end

    return Xgeo, Xgshe
end


"""
    solve_decreasing(
        Xmax::Array{<:Real, 3},
        geometry::Geometry{<:Real},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
        verbose::Bool=true,
    )

Solve GSHE trajectories for a given configuration. Loops over the geodesics. The output is
the geodesic results of shape (Nsols, 6) and the GSHE results of shape (Nsols, 2, Nϵs, 6).
"""
function solve_decreasing(
    Xmax::Array{<:Real, 3},
    geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)
    Ngeos = size(Xmax, 1)
    Xgshes = zeros(geometry.dtype, Ngeos, 2, length(ϵs), size(Xmax)[end])
    Xgeos = zeros(geometry.dtype, 2, size(Xmax)[end])


    for n in 1:Ngeos
        verbose && println("n = $n"); flush(stdout)

        Xgeo, Xgshe = solve_decreasing(Xmax[n, ..], geometry, ϵs; verbose=verbose)
        Xgeos[n, ..] .= Xgeo
        Xgshes[n, ..] .= Xgshe
    end

    return Xgeos, Xgshes
end


"""
    solve_decreasing(
        Xmaxs::Array{<:Real, 4},
        geometries::Vector{<:Geometry{<:Real}},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        configuration_verbose::Bool=true,
        perturbation_verbose::Bool=true,
    )

Solve GSHE trajectories for a vector of configurations. Returns two arrays that correspond to
the geodesic and GSHE solutions, respectively, of shape (Nconfigurations, Nsols, 6) and
(Nconfigurations, Nsols, 2, Nϵs, 6).
"""
function solve_decreasing(
    Xmaxs::Array{<:Real, 4},
    geometries::Vector{<:Geometry{<:Real}},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    configuration_verbose::Bool=true,
    perturbation_verbose::Bool=true,
)
    Nconfs = size(Xmaxs, 1)
    @assert Nconfs === length(geometries) ("`Xmaxs` and `geometries` must have the same length.")
    check_geometry_dtypes(geometries)
    Xgshes = Vector{Array{geometries[1].dtype, 4}}(undef, Nconfs)
    Xgeos = Vector{Matrix{geometries[1].dtype}}(undef, Nconfs)

    configuration_verbose && println("Solving GSHE for $Nconfs configurations."); flush(stdout)

    Threads.@threads for i in shuffled_iterators(Nconfs)
        configuration_verbose && println("Solving GSHE for geometry $i/$Nconfs"); flush(stdout)

        Xgeos[i], Xgshes[i] = solve_decreasing(Xmaxs[i, ..], geometries[i], ϵs; verbose=perturbation_verbose)
    end

    return toarray(Xgeos), toarray(Xgshes)
end


"""
    solve_increasing(
        Xgeo::Vector{<:Real},
        geometry::Geometry{<:Real},
        s::Integer,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
        verbose::Bool=true,
    )

Solve GSHE trajectories of a single geodesic and a given polarisation, starting from the geodesic.
The output shape is (Nϵs, 6).
"""
function solve_increasing(
    Xgeo::Vector{<:Real},
    geometry::Geometry{<:Real},
    s::Integer,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)
    @assert is_strictly_increasing(ϵs) "`ϵs` must be strictly increasing."
    Nϵs = length(ϵs)
    Xgshe = fill!(zeros(geometry.dtype, Nϵs, length(Xgeo)), NaN)
    nloops = Xgeo[6]

    for (i, ϵ) in enumerate(ϵs)
        verbose && println("$(round(i / Nϵs * 100, digits=2))%, ϵ=$(round(ϵ, sigdigits=2, base=10))"); flush(stdout)

        # Loop over the previously found solutions in reverse. Look up to the previous
        # 5 solutions.
        for k in reverse(max(i - 5, 1):i)
            # If previous GSHE solution available and is not NaN set it as
            # initial direction.
            if k > 1
                p0 = Xgshe[k - 1, 1:2]
                if ~any(isnan.(p0))
                    Xgshe[i, :] .= find_consecutive_minimum(geometry, ϵ, s, p0, ϵs[k - 1], nloops, Xgshe[k - 1, 8])
                    break
                end
            end

            # For the first GSHE set the geodesic solution as initial direction
            if k == 1
                Xgshe[i, :] .= find_consecutive_minimum(geometry, ϵ, s, Xgeo[1:2], 0, nloops, Xgeo[8])
            end
        end

    end

    if geometry.postproc_options.check_sols
        check_solutions!(Xgshe, Xgeo, s, geometry, ϵs, true)
    end

    return Xgshe
end


"""
    solve_increasing(
        Xgeo::Vector{<:Real},
        geometry::Geometry{<:Real},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
        verbose::Bool=true,
    )

Solve GSHE trajectories of a single geodesic and both polarisations, starting from the geodesic.
The output shape is (2, Nϵs, 6).
"""
function solve_increasing(
    Xgeo::Vector{<:Real},
    geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)
    @assert is_strictly_increasing(ϵs) "`ϵs` must be strictly increasing."
    Xgshe = zeros(geometry.dtype, 2, length(ϵs), length(Xgeo))

    for (j, s) in enumerate([+geometry.s, -geometry.s])
        Xgshe[j, ..] = solve_increasing(Xgeo, geometry, s, ϵs; verbose=verbose)
    end

    return Xgshe
end


"""
    solve_increasing(
        Xgeo::Matrix{<:Real},
        geometry::Geometry{<:Real},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
        verbose::Bool=true,
    )

Solve GSHE trajectories of a given configuration, iterating over geodesics and starting from
them. The output shape is (Nsols, s = ± 2, Nϵs, 6)
"""
function solve_increasing(
    Xgeo::Matrix{<:Real},
    geometry::Geometry{<:Real},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}};
    verbose::Bool=true,
)
    Nϵs = length(ϵs)
    Ngeos = size(Xgeo)[1]
    Xgshe = zeros(geometry.dtype, Ngeos, 2, Nϵs, size(Xgeo, 2))

    for n in 1:Ngeos
        verbose && println("n = $n"); flush(stdout)

        Xgshe[n, ..] .= solve_increasing(Xgeo[n, :], geometry, ϵs; verbose=verbose)
    end

    return Xgshe
end


"""
    solve_increasing(
        Xgeos::Array{<:Matrix{<:Real}, 3},
        geometries::Vector{<:Geometry{<:Real}},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        configuration_verbose::Bool=true,
        perturbation_verbose::Bool=true,
    )


Solve the GSHE trajectories, iterating over configurations and their geodesics. Output shape
is (Nconfigurations, Nsols, s = ± 2, Nϵs, 6).
"""
function solve_increasing(
    Xgeos::Array{<:Real, 3},
    geometries::Vector{<:Geometry{<:Real}},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    configuration_verbose::Bool=true,
    perturbation_verbose::Bool=true,
)
    Nconfs = size(Xgeos, 1)
    @assert Nconfs === length(geometries) ("`Xgeos` and `geometries` must have the same length.")
    check_geometry_dtypes(geometries)
    Xgshes = Vector{Array{geometries[1].dtype, 4}}(undef, Nconfs)

    configuration_verbose && println("Solving GSHE for $Nconfs configurations."); flush(stdout)

    Threads.@threads for i in shuffled_iterators(Nconfs)
        configuration_verbose && println("Solving GSHE for geometry $i/$Nconfs"); flush(stdout)

        Xgshes[i] = solve_increasing(Xgeos[i, ..], geometries[i], ϵs; verbose=perturbation_verbose)
    end

    return toarray(Xgshes)
end


"""
    solve_full(
        geometry::Geometry,
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        increasing_ϵ::Bool,
        Nsols::Integer;
        perturbation_verbose::Bool=true,
    )

Solve the geodesic and GSHE trajectories for a geometry. Can either solve from
the geodesic upwards or from the maximum ϵ downwards.
"""
function solve_full(
    geometry::Geometry,
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    increasing_ϵ::Bool,
    Nsols::Integer;
    perturbation_verbose::Bool=true,
)
    @assert is_strictly_increasing(ϵs) "`ϵs` must be strictly increasing."

    X0 = solve_initial(geometry, increasing_ϵ ? 0 : ϵs[end], Nsols)
    # In case of any NaNs initially just return NaN
    if any(isnan.(X0[.., 3]))
        return X0, fill(NaN, Nsols, 2, length(ϵs), 9)
    end

    # Calculate the whole thing
    if increasing_ϵ
        Xgeo = X0
        Xgshe = solve_increasing(Xgeo, geometry, ϵs; verbose=perturbation_verbose)
    else
        Xgeo, Xgshe = solve_decreasing(X0, geometry, ϵs; verbose=perturbation_verbose)
    end

    return Xgeo, Xgshe
end


"""
    solve_full(
        geometries::Vector{<:Geometry{<:Real}},
        ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
        increasing_ϵ::Bool,
        Nsols::Integer,
        tosort::Bool=true;
        configuration_verbose::Bool=true,
        perturbation_verbose::Bool=true,
    )

Solve the geodesic and GSHE trajectories for a vector of geometries. Can either solve from
the geodesic upwards or from the maximum ϵ downwards.
"""
function solve_full(
    geometries::Vector{<:Geometry{<:Real}},
    ϵs::Union{Vector{<:Real}, LinRange{<:Real}},
    increasing_ϵ::Bool,
    Nsols::Integer,
    tosort::Bool=true;
    configuration_verbose::Bool=true,
    perturbation_verbose::Bool=true,
)
    @assert is_strictly_increasing(ϵs) "`ϵs` must be strictly increasing."
    check_geometry_dtypes(geometries)
    Nconfs = length(geometries)

    # Multiprocess
    Xgeos= Vector{Matrix{geometries[1].dtype}}(undef, Nconfs)
    Xgshes = Vector{Array{geometries[1].dtype, 4}}(undef, Nconfs)
    Threads.@threads for i in shuffled_iterators(Nconfs)
        # Optionally print a message
        configuration_verbose ? (println("Solving GSHE for geometry $i/$Nconfs"); flush(stdout)) : nothing
        Xgeos[i], Xgshes[i] = solve_full(geometries[i], ϵs, increasing_ϵ, Nsols; perturbation_verbose=perturbation_verbose)
    end

    # Put into arrays and optionally sort
    Xgeos = toarray(Xgeos)
    Xgshes = toarray(Xgshes)
    tosort && Nsols > 1 && sort_configurations!(Xgeos, Xgshes)

    return Xgeos, Xgshes
end


"""
    shuffled_iterators(N::Integer)

Get a shuffled iterator 1:N.
"""
function shuffled_iterators(N::Integer)
    # Shuffle workers jobs
    if Threads.nthreads() > 1
        return shuffle!([i for i in 1:N])
    else
        return 1:Nconfs
    end
end
