import Clustering: kmeans

"""
    cartesian_to_spherical(X::Vector{Float64}})

Calculate the spherical coordinates (r, theta, phi) from Cartesian (x, y,z).
If `unit_vector` is true returns (theta, phi)
"""
function cartesian_to_spherical(X::Vector; unit_vector::Bool=false)
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
    spherical_to_cartesian2(X::Vector{Float64}; unit_vector::Bool=false)

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
    uniform_sample_sphere()

Sample a uniform point on a sphere. Returns (theta, phi), such that
0<=theta<=pi and 0<=phi<2pi.
"""
function uniform_sample_sphere()
    theta, phi = rand(2)
    theta = asin(2*(theta - 0.5)) + pi/2
    phi *= 2pi
    return [theta, phi]
end


"""
    angdist(solution, geometry::GWBirefringence.geometry, rtol::Float64=1e-8)

Calculate the angular distance between the geodesic solution and the observer.
"""
function angdist(solution, geometry::GWBirefringence.geometry,
                 rtol::Float64=1e-10)
    @unpack r, theta, phi = geometry.observer
    # Check that the radii agree within tolerance
    if ~isapprox(solution[2, end], r, rtol=rtol)
        return Inf
    end

    return angdist(solution[3:4, end], [theta, phi])
end


"""
    angdist(X1::Vector, X2::Vector)

Calculate the angular distance X1 and X2. If length of X1 and X2 is 2 assumes
inputs are `(theta, phi)` such that 0 <= theta <= pi and 0 <= phi < 2pi.
"""
function angdist(X1::Vector, X2::Vector)
    @assert length(X1) == length(X2) "Vector dimensions do not match"

    X1 = convert(Vector{BigFloat}, X1)
    X2 = convert(Vector{BigFloat}, X2)

    if length(X1) == 3
        X1 = cartesian_to_spherical(X1; unit_vector=true)
        X2 = cartesian_to_spherical(X2; unit_vector=true)
    end

    dist = acos(cos(X1[1]) * cos(X2[1])
                + sin(X1[1]) * sin(X2[1]) * cos(X1[2] - X2[2]))

    return Float64(dist)
end


"""
    assign_clusters(X::Matrix, tol::Float64=1e-10)

Assign vectors in X of shape (Nsamples, ndim) to clusters using k-means.
Searches for clusters until the cost function is below tolerance and returns
cluster assignemnts.
"""
function assign_clusters(X::Matrix, tol::Float64=1e-10)
    Xcart = mapslices(spherical_to_cartesian, X, dims=(2))

    for k in 1:size(X)[1]
        R = kmeans(transpose(Xcart), k; maxiter=500)
        @assert R.converged "Kmeans convergence"
        
        if sum(R.costs) < tol
            return R.assignments
        end
    end
end
