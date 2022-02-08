"""
    grid_func(func::Function, thetas::T=LinRange(0, pi, 50),
              phis::T=LinRange(0, 2pi, 50)) where T<:Union{Vector, LinRange}

Evaluate `func` on a grid given by `thetas` and `phis`.
"""
function grid_func(func::Function, thetas::T=LinRange(0, pi, 50),
                   phis::T=LinRange(0, 2pi, 50)) where T<:Union{Vector, LinRange}
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
