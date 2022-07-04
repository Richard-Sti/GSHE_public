"""
    azimuthal_angle(y::Real, x::Real)

Convert Cartesian coordinates `y` and `x` into an azimuthal angle starting at `x`, ensuring
it is within [0, 2π).
"""
function azimuthal_angle(y::Real, x::Real)
    ϕ = atan(y, x)
    if ϕ < 0
        return ϕ + 2π
    end
    return ϕ
end


"""
    cartesian_to_spherical(X::Vector{<:Real})

Calculate the spherical coordinates (r, θ, ϕ) from Cartesian (x, y,z). TODO: add Kerr spin
"""
function cartesian_to_spherical(X::Vector{<:Real})
    x, y, z = X
    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    ϕ = azimuthal_angle(y, x)
    return [r, θ, ϕ]
end


function cartesian_to_spherical!(X::Vector{<:Real})
    x, y, z  = X
    X[1] = sqrt(x^2 + y^2 + z^2)
    X[2] = acos(z / r)
    X[3] = azimuthal_angle(y, x)
end


"""
    spherical_to_cartesian(X::Vector{<:Real}, a::Real=0.0)

Calculate the Cartesian coordinates (x, y, z) from spherical (θ, ϕ) or (r, θ, ϕ), depending
on the input length, `a` is the Kerr spin parameter.
"""
function spherical_to_cartesian(X::Vector{<:Real}, a::Real=0.0)
    if length(X) == 2
        θ, ϕ = X
        r = 1
    else
        r, θ, ϕ = X
    end
    sθ = sin(θ)
    radius = sqrt(r^2 + a^2)
    x = radius * cos(ϕ) * sθ
    y = radius * sin(ϕ) * sθ
    z = radius * cos(θ)
    return [x, y, z]
end


function spherical_to_cartesian!(X::Vector{<:Real}, a::Real=0.0)
    if length(X) == 2
        θ, ϕ = X
        r = 1
    else
        r, θ, ϕ = X
    end
    sθ = sin(θ)
    radius = sqrt(r^2 + a^2)
    X[1] = radius * cos(ϕ) * sθ
    X[2] = radius * sin(ϕ) * sθ
    X[3] = radius * cos(θ)
end


"""
    rvs_sphere(θmax::Real=π; dtype::DataType=Float64)

Sample a uniform point on a sphere. Returns (θ, ϕ), such that 0 ≤ theta ≤ π , 0 ≤ ϕ  < 2π.
If θmax ≤ π point will be sampled within θmax of the north pole.
"""
function rvs_sphere(θmax::Real=π; dtype::DataType=Float64)
    # Sample within [0, 1] uniformly
    sample = rand(dtype, 2)
    # if θmax is not π restrict it to a smaller range
    if θmax < π
        sample[1] *= (1 - cos(θmax)) / 2
    end
    # Get the actual angles
    sample[1] = asin(2*(sample[1] - 0.5)) + π/2
    sample[2] *= 2π
    return sample
end


function rvs_sphere_restricted(Δθ::Real=0.0, Δϕ::Real=0.0; dtype::DataType=Float64)
    sample = rand(dtype, 2)

    sample[1] = acos(2*(sample[1] - 0.5) * abs(cos(Δθ)))
    sample[2] = sample[2] * (2π - 2*Δϕ) + Δϕ
    return sample
end

function rvs_sphere_restricted(N::Integer, Δθ::Real=0.0, Δϕ::Real=0.0; dtype::DataType=Float64)
    return mapreduce(permutedims, vcat, [rvs_sphere_restricted(Δθ, Δϕ; dtype=dtype) for i in 1:N])
end


"""
    rvs_sphere_y(θmax::Real=π; dtype::DataType=Float64)

Sample a point (θ, ϕ) within distance θmax of the y-axis (0, 1, 0).
"""
function rvs_sphere_y(θmax::Real=π; dtype::DataType=Float64)
    sample = rvs_sphere(θmax; dtype=dtype)

    θ, ϕ = sample
    sample[1] = acos(-sin(θ)*cos(ϕ))
    sample[2] = atan(sin(θ)*sin(ϕ), cos(θ)) + π/2
    sample[2] < 0 ? (sample[2] += 2π) : nothing
    return sample
end


"""
    rotate_to_y(x::Vector{<:Real}, p::Vector{<:Real})

Rotate `x` with a rotation that moves `p` to the y-axis. All vectors assumed to be given in
spherical coordinates (θ, ϕ).
"""
function rotate_to_y(x::Vector{<:Real}, p::Vector{<:Real})
    ψ, ρ = p
    θ, ϕ = x
    # Check data dtypes
    dtype = typeof(ψ)
    @assert all(isa(a, dtype) for a in [ρ, θ, ϕ]) "Inputs have mixed data dtypes."

    sψ, cψ = sin(ψ), cos(ψ)
    sθ, cθ = sin(θ), cos(θ)
    c_azim = cos(ρ - ϕ)
    out = dtype[0.0, 0.0]
    out[1] = acos(-c_azim * cψ * sθ + cθ * sψ)
    out[2] = azimuthal_angle(cθ * cψ + c_azim * sθ * sψ, sθ * sin(ρ - ϕ))
    return out
end


function rotate_to_y!(x::Vector{<:Real}, p::Vector{<:Real})
    ψ, ρ = p
    θ, ϕ = x
    sψ, cψ = sin(ψ), cos(ψ)
    sθ, cθ = sin(θ), cos(θ)
    c_azim = cos(ρ - ϕ)
    x[1] = acos(-c_azim * cψ * sθ + cθ * sψ)
    x[2] = azimuthal_angle(cθ * cψ + c_azim * sθ * sψ, sθ * sin(ρ - ϕ))
end


"""
    rotate_from_y(x::Vector{<:Real}, p::Vector{<:Real})

Rotate `x` with an inverse of a rotation that moves `p` to the y-axis. All vectors assumed
to be given in spherical coordinates (θ, ϕ).
"""
function rotate_from_y(x::Vector{<:Real}, p::Vector{<:Real})
    ψ, ρ = p
    θ, ϕ = x

    # Check data dtypes
    dtype = typeof(ψ)
    @assert all(isa(a, dtype) for a in [ρ, θ, ϕ]) "Inputs have mixed data dtypes."

    sψ, cψ = sin(ψ), cos(ψ)
    sρ, cρ = sin(ρ), cos(ρ)
    sθ, cθ = sin(θ), cos(θ)
    sϕ, cϕ = sin(ϕ), cos(ϕ)

    out = dtype[0.0, 0.0]
    out[1] = acos(cψ*sθ*sϕ + cθ*sψ)
    out[2] = azimuthal_angle(
        -cρ*cϕ*sθ + sρ*(-cθ*cψ + sθ*sϕ*sψ),
        -cθ*cρ*cψ + sθ*(cϕ*sρ + cρ*sϕ*sψ))
    return out
end


function rotate_from_y!(x::Vector{<:Real}, p::Vector{<:Real})
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
    angdist(X1::Vector{<:Real}, X2::Vector{<:Real})
Calculate the angular distance between X1 = (θ1, ϕ1) and X2 = (θ2,ϕ2).
"""
function angdist(X1::Vector{<:Real}, X2::Vector{<:Real})
    θ1, ϕ1 = X1
    θ2, ϕ2 = X2
    return angdist(θ1, ϕ1, θ2, ϕ2)
end


function angdist(θ1::Real, ϕ1::Real, θ2::Real, ϕ2::Real)
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
    atan_transform(x::Real, α::Real=π/2)

Transform `x` according to f(x) = π / 2 + α / (π / 2) * atan(x)
"""
function atan_transform(x::Real, α::Real=π/2)
    return π / 2 +  α / (π / 2) * atan(x)
end


"""
    atan_invtransform(y::Real, alpha::Real=π/2)

Inverse transformation of `atan_transform` defined above.
"""
function atan_invtransform(y::Real, α::Real=π/2)
    return tan((π / 2) / α * (y - π / 2))
end


"""
    shadow2angle(ks::Vector{<:Real})

Convert the shadow k2 and k3 coordinates to ψ and ρ
"""
function shadow2angle(ks::Vector{<:Real})
    k2, k3 = ks
    ψ = acos(k3)
    ρ = π + asin(k2 / sqrt(1 - k3^2))
    return [ψ, ρ]
end


"""
    shadow2angle(ks::Vector{<:Real})

Convert the ingoing ψ and ρ coordinates to k2 and k3.
"""
function angle2shadow(X::Vector{<:Real})
    ψ, ρ = X
    k3 = cos(ψ)
    k2 = sin(ψ) * sin(ρ)
    return [k2, k3]
end