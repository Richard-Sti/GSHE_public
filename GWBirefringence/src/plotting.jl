using LaTeXStrings
import Plots


function plot_initial_conditions(
    Xspinhall::Array{Float64, 4},
    Xgeo::Matrix{Float64},
    igeo::Int64
)
    fig = Plots.plot(dpi=600, xlabel=L"\rho", ylabel=L"\psi")
    for s in [2, -2]
        x = @view Xspinhall[:, s == 2 ? 1 : 2, igeo, 1:2]
        Plots.scatter!(fig, x[:, 2], x[:, 1], label="s=$s")
    end
    Plots.scatter!(fig, [Xgeo[igeo, 2]], [Xgeo[igeo, 1]], label="Geodesic")
    return fig
end


function plot_arrival_times(
    ϵs::Union{Vector{Float64}, LinRange{Float64}},
    Xspinhall::Array{Float64, 4},
    Xgeo::Matrix{Float64},
    igeo::Int64
)   
    fig = Plots.plot(xaxis=:log, xlabel=L"\epsilon", ylabel=L"t_{\rm obs}",
                     legend=:topleft)
    xplus = @view Xspinhall[:, 1, igeo, 3]
    xminus = @view Xspinhall[:, 2, igeo, 3]

    Plots.scatter!(fig, ϵs, xplus, label=L"s=2")
    Plots.scatter!(fig, ϵs, xminus, label=L"s=-2")
    Plots.hline!(fig, [Xgeo[igeo, 3]], label=L"{\rm Geodesic}")
    return fig
end

function plot_time_difference(
    ϵs::Union{Vector{Float64}, LinRange{Float64}},
    Xspinhall::Array{Float64, 4},
    Xgeo::Matrix{Float64}
)
    Nsols = size(Xgeo)[1]
    fig = Plots.plot(
        xaxis=:log, yaxis=:log, legend=:topleft, dpi=600,
        xlabel=L"\epsilon",
        ylabel=L"|t_{\rm obs}(s=2) - t_{\rm obs}(s=-2)|")
    for igeo in 1:Nsols
        xplus = @view Xspinhall[:, 1, igeo, 3]
        xminus = @view Xspinhall[:, 2, igeo, 3]

        Plots.scatter!(fig, ϵs, abs.(xplus - xminus), label="Geodesic $igeo")
    end
    return fig
end


"""
    grid_func(func::Function, thetas::T=LinRange(0, pi, 50),
              phis::T=LinRange(0, 2pi, 50)) where T<:Union{Vector, LinRange}

Evaluate `func` on a grid given by `thetas` and `phis` if possible attempts
to use multiple threads.
"""
function grid_func(func::Function, thetas::T=LinRange(0, pi, 50),
                   phis::T=LinRange(0, 2pi, 50)) where T<:Union{Vector,
                                                                LinRange}
    N = length(thetas) * length(phis)
    grid = zeros(N, 2)
    Z = zeros(N)
    k = 1
    # Create a meshgrid
    for phi in phis
        for theta in thetas
            grid[k, 1] = theta
            grid[k, 2] = phi
            k += 1
        end
    end
    # Calculate fmin in parallel
    Threads.@threads for i in 1:N
        Z[i] = func([grid[i, 1], grid[i, 2]])
    end
    
    return grid, Z
end
