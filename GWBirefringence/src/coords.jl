"""
    cartesian_to_spherical(X::Vector{Float64}})

Calculate the spherical coordinates (r, theta, phi) from Cartesian (x, y,z).
"""
function cartesian_to_spherical(X::Vector{Float64})
    x, y, z = X
    r = sqrt(x^2 + y^2 + z^2)
    theta = acos(z / r)
    phi = atan(y, x)
    return [r, theta, phi]
end


"""
    spherical_to_cartesian2(X::Vector{Float64}; unit_vector::Bool=false)

Calculate the Cartesian coordinates (x, y, z) from spherical (r, theta, phi).
"""
function spherical_to_cartesian(X::Vector{Float64}; unit_vector::Bool=false)
    r, theta, phi = X
    x = cos(phi) * sin(theta)
    y = sin(phi) * sin(theta)
    z = cos(theta)
    if unit_vector
        [x, y, z]
    else
        r.* [x, y, z]
    end
end
