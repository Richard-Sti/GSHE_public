"""
    cartesian_to_spherical(X::Vector{GWFloat}; unit_vector::Bool=false)

Calculate the spherical coordinates (r, theta, phi) from Cartesian (x, y,z).
If `unit_vector` is true returns (theta, phi)
"""
function cartesian_to_spherical(X::Vector;
                                unit_vector::Bool=false)
    x, y, z = X
    r = sqrt(x^2 + y^2 + z^2)
    theta = acos(z / r)
    phi = atan(y, x) + pi
    if unit_vector
        return [theta, phi]
    else
        return [r, theta, phi]
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
    X[3] = atan(y, x) + pi
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
    uniform_sample_sphere(return_cartesian::Bool=false)

Sample a uniform point on a sphere. Returns (theta, phi), such that
0<=theta<=pi and 0<=phi<2pi.
"""
function uniform_sample_sphere(return_cartesian::Bool=false)
    theta, phi = rand(GWFloat, 2)
    theta = asin(2*(theta - 0.5)) + pi/2
    phi *= 2pi

    if return_cartesian
        return spherical_to_cartesian([theta, phi])
    end

    return [theta, phi]
end


"""
    angdist(solution, geometry::GWBirefringence.geometry, rtol::Float64=1e-10)

Calculate the angular distance between the geodesic solution and the observer.
"""
function angdist(solution, geometry::Geometry,
                 rtol::Float64=1e-10)
    @unpack r, theta, phi = geometry.observer
    # Check that the radii agree within tolerance
    if ~isapprox(solution[2, end], r, rtol=rtol)
        return Inf
    end

    return angdist(solution[3:4, end], [theta, phi])
end


"""
    angdist(X1::Vector{GWFloat}, X2::Vector{GWFloat})

Calculate the angular distance between X1 and X2.
"""
function angdist(X1::Vector, X2::Vector)

    dphi = X1[2] - X2[2]
    stheta1, stheta2 = sin(X1[1]), sin(X2[1])
    ctheta1, ctheta2 = cos(X1[1]), cos(X2[1])
    cdphi = cos(dphi)

    x = begin
        x1 = (stheta2 * sin(dphi))^2
        x2 = (stheta1 * ctheta2 - ctheta1 * stheta2 * cdphi)^2
        sqrt(x1 + x2)
    end

    y = ctheta1 * ctheta2 + stheta1 * stheta2 * cdphi

    return atan(x, y)
end
