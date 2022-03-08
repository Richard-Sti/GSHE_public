"""
    cartesian_to_spherical(X::Vector{GWFloat}; unit_vector::Bool=false)

Calculate the spherical coordinates (r, theta, phi) from Cartesian (x, y,z).
If `unit_vector` is true returns (theta, phi)
"""
function cartesian_to_spherical(
    X::Vector;
    unit_vector::Bool=false
)
    # Unpack the vector
    x, y, z = X
    # Do the transformation
    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    ϕ = atan(y, x)
    # Correct the ϕ range
    ϕ < 0 ? (ϕ += 2pi) : nothing
    if unit_vector
        return [θ, ϕ]
    else
        return [r, θ, ϕ]
    end
end


"""
    cartesian_to_spherical(X::Vector{GWFloat})

Calculate the spherical coordinates (r, theta, phi) from Cartesian (x, y,z).
"""
function cartesian_to_spherical!(X::Vector)
    x, y, z = X
    r = sqrt(x^2 + y^2 + z^2)
    X[1] = r
    X[2]= acos(z / r)
    X[3] = atan(y, x)
    # Correct the ϕ range
    X[3] < 0 ? (X[3] += 2pi) : nothing
end


"""
    spherical_to_cartesian(X::Vector)

Calculate the Cartesian coordinates (x, y, z) from spherical (theta, phi) or
(r, theta, phi), depending on the input length.
"""
function spherical_to_cartesian(X::Vector)
    dim = length(X)
    if dim  == 2
        theta, phi = X
    else
        r, theta, phi = X
    end

    x = cos(phi) * sin(theta)
    y = sin(phi) * sin(theta)
    z = cos(theta)
    if dim == 2
        return [x, y, z]
    else
        return r .* [x, y, z]
    end
end


"""
    uniform_sample_sphere(
        return_cartesian::Bool=false,
        θmax::Union{GWFloat, Irrational}=π
    )

Sample a uniform point on a sphere. Returns (θ, ϕ), such that
0 ≤ theta ≤ π and 0 ≤ ϕ  < 2π. If `return_cartesian` is true returns the
Cartesian coordinates of the point.
"""

function uniform_sample_sphere(
    return_cartesian::Bool=false,
    θmax::GWFloat=1π
)
    # Sample within [0, 1] uniformly
    θ, ϕ = rand(GWFloat, 2)
    # if θmax is not π restrict it to a smaller range
    if θmax != π
        θ *= (1 - cos(θmax)) / 2
    end
    # Get the actual angles
    θ = asin(2*(θ - 0.5)) + π/2
    ϕ *= 2π

    if return_cartesian
        return spherical_to_cartesian([θ, ϕ])
    end

    return [θ, ϕ]
end


"""
    uniform_sample_sphere_near_x(θmax::GWFloat=1π)

Sample a point within distance θmax of the y-axis.
"""
function uniform_sample_sphere_near_y(θmax::GWFloat=1π)
    θ_0, ϕ_0 = GWBirefringence.uniform_sample_sphere(false, θmax)

    θ = acos(-sin(θ_0)*cos(ϕ_0))
    ϕ = atan(sin(θ_0)*sin(ϕ_0), cos(θ_0)) + π/2
    ϕ < 0 ? (ϕ += 2pi) : nothing
    return [θ, ϕ]
end


"""
    rotation_to_y(θ::GWFloat, ϕ::GWFloat)

Calculate the rotation matrix to rotate a unit vector given by θ and ϕ to
lie on the y-axis.
"""
function rotation_to_y(θ::GWFloat, ϕ::GWFloat)
    sθ, cθ = sin(θ), cos(θ)
    sϕ, cϕ = sin(ϕ), cos(ϕ)
    return [ sϕ    -cϕ    0.0
             sθ*cϕ  sθ*sϕ cθ
            -cθ*cϕ -cθ*sϕ sθ]
end


"""
    angdist(X1::Vector{GWFloat}, X2::Vector{GWFloat})

Calculate the angular distance between X1 and X2.
"""
function angdist(X1::Vector{GWFloat}, X2::Vector{GWFloat})
    dϕ = X1[2] - X2[2]
    sθ1, sθ2 = sin(X1[1]), sin(X2[1])
    cθ1, cθ2 = cos(X1[1]), cos(X2[1])
    cdϕ = cos(dϕ)

    x = begin
        x1 = (sθ2 * sin(dϕ))^2
        x2 = (sθ1 * cθ2 - cθ1 * sθ2 * cdϕ)^2
        sqrt(x1 + x2)
    end

    y = cθ1 * cθ2 + sθ1 * sθ2 * cdϕ

    return atan(x, y)
end
