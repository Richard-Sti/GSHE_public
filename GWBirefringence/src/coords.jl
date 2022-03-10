"""
    azimuthal_angle(y::GWFloat, x::GWFloat)

Convert Cartesian coordinates `y` and `x` into an azimuthal angle starting at
`x`, ensuring it is within [0, 2π).
"""
function azimuthal_angle(y::GWFloat, x::GWFloat)
    ϕ = atan(y, x)
    if ϕ < 0
        return ϕ + 2π
    end
    return ϕ
end


"""
    cartesian_to_spherical(X::Vector{GWFloat})

Calculate the spherical coordinates (r, θ, ϕ) from Cartesian (x, y,z).
"""
function cartesian_to_spherical!(X::Vector{GWFloat})
    x, y, z  = X
    X[1] = sqrt(x^2 + y^2 + z^2)
    X[2] = acos(z / r)
    X[3] = azimuthal_angle(y, x)
end


"""
    cartesian_to_spherical(X::Vector{GWFloat})

Calculate the spherical coordinates (r, θ, ϕ) from Cartesian (x, y,z).
"""
function cartesian_to_spherical(X::Vector{GWFloat})
    x, y, z = X
    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    ϕ = azimuthal_angle(y, x)
    return [r, θ, ϕ]
end


"""
    spherical_to_cartesian(X::Vector)

Calculate the Cartesian coordinates (x, y, z) from spherical (θ, ϕ) or
(r, θ, ϕ), depending on the input length.
"""
function spherical_to_cartesian!(X::Vector{GWFloat})
    if length(X) == 2
        θ, ϕ = X
        r = 1.0
    else
        r, θ, ϕ = X
    end
    sθ, cθ = sin(θ), cos(θ)
    X[1] = r * cos(ϕ) * sθ
    X[2] = r * sin(ϕ) * sθ
    X[3] = r * cθ
end


"""
    spherical_to_cartesian(X::Vector{GWFloat})

Calculate the Cartesian coordinates (x, y, z) from spherical (θ, ϕ) or
(r, θ, ϕ), depending on the input length.
"""
function spherical_to_cartesian(X::Vector{GWFloat})
    if length(X) == 2
        θ, ϕ = X
        r = 1.0
    else
        r, θ, ϕ = X
    end
    r, sθ, cθ = r, sin(θ), cos(θ)
    x = r * cos(ϕ) * sθ
    y = r * sin(ϕ) * sθ
    z = r * cθ
    return [x, y, z]
end


"""
    rvs_sphere(θmax::GWFloat=1π)

Sample a uniform point on a sphere. Returns (θ, ϕ), such that 0 ≤ theta ≤ π and
0 ≤ ϕ  < 2π. If θmax ≤ π point will be sampled within θmax of the north pole.
"""
function rvs_sphere(θmax::GWFloat=1π)
    # Sample within [0, 1] uniformly
    sample = rand(GWFloat, 2)
    # if θmax is not π restrict it to a smaller range
    if θmax < π
        sample[1] *= (1 - cos(θmax)) / 2
    end
    # Get the actual angles
    sample[1] = asin(2*(sample[1] - 0.5)) + π/2
    sample[2] *= 2π
    return sample
end


"""
    rvs_sphere_y(θmax::GWFloat=1π)

Sample a point (θ, ϕ) within distance θmax of the y-axis (0, 1, 0).
"""
function rvs_sphere_y(θmax::GWFloat=1π)
    sample = rvs_sphere(θmax)

    θ, ϕ = sample
    sample[1] = acos(-sin(θ)*cos(ϕ))
    sample[2] = atan(sin(θ)*sin(ϕ), cos(θ)) + π/2
    sample[2] < 0 ? (sample[2] += 2π) : nothing
    return sample
end


"""
    rotate_to_y(x::Vector{GWFloat}, p::Vector{GWFloat})

Rotate `x` with a rotation that moves `p` to the y-axis. All vectors assumed
to be given in spherical coordinates (θ, ϕ).
"""
function rotate_to_y(x::Vector{GWFloat}, p::Vector{GWFloat})
    ψ, ρ = p
    θ, ϕ = x
    sψ, cψ = sin(ψ), cos(ψ)
    sθ, cθ = sin(θ), cos(θ)
    c_azim = cos(ρ - ϕ)
    out = [0.0, 0.0]
    out[1] = acos(-c_azim * cψ * sθ + cθ * sψ)
    out[2] = azimuthal_angle(cθ * cψ + c_azim * sθ * sψ, sθ * sin(ρ - ϕ))
    return out
end


"""
    rotate_to_y!(x::Vector{GWFloat}, p::Vector{GWFloat})

Rotate `x` with a rotation that moves `p` to the y-axis. All vectors assumed
to be given in spherical coordinates (θ, ϕ).
"""
function rotate_to_y!(x::Vector{GWFloat}, p::Vector{GWFloat})
    ψ, ρ = p
    θ, ϕ = x
    sψ, cψ = sin(ψ), cos(ψ)
    sθ, cθ = sin(θ), cos(θ)
    c_azim = cos(ρ - ϕ)
    x[1] = acos(-c_azim * cψ * sθ + cθ * sψ)
    x[2] = azimuthal_angle(cθ * cψ + c_azim * sθ * sψ, sθ * sin(ρ - ϕ))
end


"""
    rotate_from_y(x::Vector{GWFloat}, p::Vector{GWFloat})

Rotate `x` with an inverse of a rotation that moves `p` to the y-axis.
All vectors assumed to be given in spherical coordinates (θ, ϕ).
"""
function rotate_from_y(x::Vector{GWFloat}, p::Vector{GWFloat})
    ψ, ρ = p
    θ, ϕ = x
    sψ, cψ = sin(ψ), cos(ψ)
    sρ, cρ = sin(ρ), cos(ρ)
    sθ, cθ = sin(θ), cos(θ)
    sϕ, cϕ = sin(ϕ), cos(ϕ)

    out = [0.0, 0.0]
    out[1] = acos(cψ*sθ*sϕ + cθ*sψ)
    out[2] = azimuthal_angle(
        -cρ*cϕ*sθ + sρ*(-cθ*cψ + sθ*sϕ*sψ),
        -cθ*cρ*cψ + sθ*(cϕ*sρ + cρ*sϕ*sψ))
    return out
end


"""
    rotate_from_y!(x::Vector{GWFloat}, p::Vector{GWFloat})

Rotate `x` with an inverse of a rotation that moves `p` to the y-axis.
All vectors assumed to be given in spherical coordinates (θ, ϕ).
"""
function rotate_from_y!(x::Vector{GWFloat}, p::Vector{GWFloat})
    ψ, ρ = p
    θ, ϕ = x
    sψ, cψ = sin(ψ), cos(ψ)
    sρ, cρ = sin(ρ), cos(ρ)
    sθ, cθ = sin(θ), cos(θ)
    sϕ, cϕ = sin(ϕ), cos(ϕ)
    x[1] = acos(cψ*sθ*sϕ + cθ*sψ)
    x[2] = azimuthal_angle(
        -cρ*cϕ*sθ + sρ*(-cθ*cψ + sθ*sϕ*sψ),
        -cθ*cρ*cψ + sθ*(cϕ*sρ + cρ*sϕ*sψ))
end


"""
    angdist(θ1::GWFloat, ϕ1::GWFloat, θ2::GWFloat, ϕ2::GWfloat)

Calculate the angular distance between (θ1, ϕ1) and (θ2, ϕ2).
"""
function angdist(θ1::GWFloat, ϕ1::GWFloat, θ2::GWFloat, ϕ2::GWFloat)
    dϕ = ϕ1 - ϕ2
    sθ1, sθ2 = sin(θ1), sin(θ2)
    cθ1, cθ2 = cos(θ1), cos(θ2)
    cdϕ = cos(dϕ)

    x = begin
        x1 = (sθ2 * sin(dϕ))^2
        x2 = (sθ1 * cθ2 - cθ1 * sθ2 * cdϕ)^2
        sqrt(x1 + x2)
    end
    y = cθ1 * cθ2 + sθ1 * sθ2 * cdϕ

    return atan(x, y)
end


"""
    angdist(X1::Vector{GWFloat}, X2::Vector{GWFloat})

Calculate the angular distance between `X1` and `X2` spherical vectors
specified as (θ, ϕ).
"""
function angdist(X1::Vector{GWFloat}, X2::Vector{GWFloat})
    θ1, ϕ1 = X1
    θ2, ϕ2 = X2
    return angdist(θ1, ϕ1, θ2, ϕ2)
end


"""
    atan_transform(x::GWFloat, α::GWFloat=π/2)

Transform `x` according to f(x) = π / 2 + α / (π / 2) * atan(x)
"""
function atan_transform(x::GWFloat, α::GWFloat=π/2)
    return π / 2 +  α / (π / 2) * atan(x)
end


"""
    atan_invtransform(y::GWFloat, alpha::GWFloat=π/2)

Inverse transformation of `atan_transform` defined above.
"""
function atan_invtransform(y::GWFloat, α::GWFloat=π/2)
    return tan((π / 2) / α * (y - π / 2))
end