"""
    pi0(ψ, ρ, r, θ, a)

Calculate the initial covectors ``p_i``.

TODO: give descriptions of the arguments following julia documentation.
"""
function pi0(ψ, ρ, r, θ, a)
    s_θ, c_θ = sin(θ), cos(θ)
    [sqrt((a.^2 .*c_θ.^2 + r.^2) ./(a.^2 + r .*(r - 2))) .*sin(ρ) .*sin(ψ), sqrt(a.^2 .*c_θ.^2 + r.^2) .*sin(ψ) .*cos(ρ), s_θ .*(-a .*s_θ .*sqrt((a.^2 + r .*(r - 2)) ./(a.^2 .*c_θ.^2 + r.^2)) + (a.^2 + r.^2) .*cos(ψ) ./sqrt(a.^2 .*c_θ.^2 + r.^2))]
end
