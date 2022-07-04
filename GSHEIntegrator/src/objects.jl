@with_kw mutable struct SphericalCoords{T <: Real} <: Number
    t::T = 0.0
    r::T
    θ::T
    ϕ::T
end

@with_kw mutable struct ODESolverOptions
    reltol::Real=1e-14
    abstol::Real=1e-14
    maxiters::Integer=10000
    interp_points::Integer=10
    Δθ::Real=0.00001
    horizon_tol::Real=1.0001
    no_loops::Bool=false
    verbose::Bool=false

end

@with_kw mutable struct OptimiserOptions
    radius_reltol::Real=1e-10
    angdist_to_old::Real=1e-9
    τ_to_old::Real=1e-9
    Ninit::Integer=30
    Nconsec::Integer=20
    loss_atol::Real=1e-12
    optim_options::Options=Options(
        iterations=1000, g_abstol=1e-14, g_reltol=1e-14, outer_g_abstol=1e-14,
        outer_g_reltol=1e-14)
    alg::NelderMead=NelderMead()
    relθmax::Real=0.13
    gshe_convergence_verbose::Bool=false
    Δσmult::Real=1.5
end

@with_kw mutable struct PostprocOptions
    integration_error::Real=1e-12
    check_sols::Bool=true
    verbose::Bool=false
    R2tol::Real=0.005
    Ncorr::Integer=1
    Nboots::Integer=1000
    minpoints::Integer=6
    expslope::Real=2
    geodesics_Δσ::Real=1e-10
    geodesics_Δt::Real=1e-9
end


@with_kw mutable struct Geometry{T <: Real}
    dtype::DataType
    source::SphericalCoords{T}
    observer::SphericalCoords{T}
    direction_coords::Symbol=:spherical
    getmagnification::Bool=false
    s::Integer = 2
    a::T
    arrival_time::T = 0.0
    redshift::T = 0.0
    nloops::T = 0.0
    ϕkilling::T = 0.0
    ode_options::ODESolverOptions=ODESolverOptions()
    opt_options::OptimiserOptions=OptimiserOptions()
    postproc_options::PostprocOptions=PostprocOptions()
end
