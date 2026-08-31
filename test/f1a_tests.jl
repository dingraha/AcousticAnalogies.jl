module F1ATests

using AcousticAnalogies

using ADTypes: ADTypes
using FLOWMath
using ForwardDiff: ForwardDiff
using LinearAlgebra: norm
using NLsolve
using Polynomials
using Test

function cf1_integrand(se, obs, t)
    c0 = se.c0

    # Need to get the retarded time.
    R(τ) = [t - (τ[1] + norm(obs(t) .- se.y0dot(τ[1]))/c0)]
    result = nlsolve(R, [-0.1], autodiff=ADTypes.AutoForwardDiff())
    if !converged(result)
        @error "nlsolve retarded time calculation did not converge:\n$(result)"
    end
    τ = result.zero[1]

    # Position of source at the retarted time.
    y = se.y0dot(τ)

    # Position vector from source to observer.
    rv = obs(t) .- y

    # Distance from source to observer.
    r = AcousticAnalogies.norm_cs_safe(rv)

    # Unit vector pointing from source to observer.
    rhat = rv./r

    # First time derivative of rv.
    rv1dot = -se.y1dot(τ)

    # Mach number of the velocity of the source in the direction of the
    # observer.
    Mr = AcousticAnalogies.dot_cs_safe(-rv1dot/se.c0, rhat)

    # Now evaluate the integrand.
    p_m_integrand = se.ρ0/(4*pi)*se.Λ*se.Δr/(r*(1 - Mr))

    # Loading at the retarded time.
    f0dot = se.f0dot(τ)

    p_d_integrand_ff = (1/(4*pi*c0))*AcousticAnalogies.dot_cs_safe(f0dot, rhat)/(r*(1 - Mr))*se.Δr
    p_d_integrand_nf = (1/(4*pi*c0))*AcousticAnalogies.dot_cs_safe(f0dot, rhat)*c0/(r^2*(1 - Mr))*se.Δr

    return τ, p_m_integrand, p_d_integrand_ff, p_d_integrand_nf
end

@testset "Compact F1A tests" begin
    # rho = 1.226  # kg/m^3
    rho = 1.226e6  # kg/m^3
    c0 = 340.0  # m/s
    Rtip = 1.1684  # meters
    radii = 0.99932*Rtip
    dradii = (0.99932 - 0.99660)*Rtip  # m
    area_over_chord_squared = 0.064
    chord = 0.47397E-02 * Rtip
    Λ = area_over_chord_squared * chord^2

    theta = 90.0*pi/180.0
    x0 = [cos(theta), 0.0, sin(theta)].*100.0.*12.0.*0.0254  # 100 ft in meters
    obs = StationaryAcousticObserver(x0)

    # Need the position and velocity of the source as a function of
    # source/retarded time. How do I want it to move? I want it to rotate around
    # an axis on the origin, pointing in the x direction.
    rpm = 2200
    omega = 2*pi/60*rpm
    period = 60/rpm
    fn = 180.66763939805125
    fc = 19.358679206883078
    y0dot(τ) = [0,  radii*cos(omega*τ), radii*sin(omega*τ)]
    y1dot(τ) = [0, -omega*radii*sin(omega*τ), omega*radii*cos(omega*τ)]
    y2dot(τ) = [0, -omega^2*radii*cos(omega*τ), -omega^2*radii*sin(omega*τ)]
    y3dot(τ) = [0, omega^3*radii*sin(omega*τ), -omega^3*radii*cos(omega*τ)]
    f0dot(τ) = [-fn, -sin(omega*τ)*fc, cos(omega*τ)*fc]
    f1dot(τ) = [0, -omega*cos(omega*τ)*fc, -omega*sin(omega*τ)*fc]
    u(τ) = y0dot(τ)./radii
    sef1 = CompactF1ASourceElement(rho, c0, dradii, Λ, y0dot, y1dot, nothing, nothing, f0dot, nothing, 0.0, u)

    t = 0.0
    dt = period*0.5^4

    τ0, pmi0, pdiff0, pdinf0 = cf1_integrand(sef1, obs, t)
    sef1a = CompactF1ASourceElement(rho, c0, dradii, Λ, y0dot(τ0), y1dot(τ0), y2dot(τ0), y3dot(τ0), f0dot(τ0), f1dot(τ0), τ0, u(τ0))
    apth = noise(sef1a, obs)

    err_prev_pm = nothing
    err_prev_pd = nothing
    dt_prev = nothing
    dt_curr = dt
    first_time = true

    err_pm = Vector{Float64}()
    err_pd = Vector{Float64}()
    dts = Vector{Float64}()
    ooa_pm = Vector{Float64}()
    ooa_pd = Vector{Float64}()
    for n in 1:7
        τ_1, pmi_1, pdiff_1, pdinf_1 = cf1_integrand(sef1, obs, t-dt_curr)
        τ1, pmi1, pdiff1, pdinf1 = cf1_integrand(sef1, obs, t+dt_curr)

        p_m_f1 = (pmi_1 - 2*pmi0 + pmi1)/(dt_curr^2)
        p_d_f1 = (pdiff1 - pdiff_1)/(2*dt_curr) + pdinf0

        err_curr_pm = abs(p_m_f1 - apth.p_m)
        err_curr_pd = abs(p_d_f1 - apth.p_d)

        if first_time
            first_time = false
        else
            push!(ooa_pm, log(err_curr_pm/err_prev_pm)/log(dt_curr/dt_prev))
            push!(ooa_pd, log(err_curr_pd/err_prev_pd)/log(dt_curr/dt_prev))
        end

        push!(dts, dt_curr)
        push!(err_pm, err_curr_pm)
        push!(err_pd, err_curr_pd)

        dt_prev = dt_curr
        err_prev_pm = err_curr_pm
        err_prev_pd = err_curr_pd
        dt_curr = 0.5*dt_curr
    end

    # Fit a line through the errors on a log-log plot, then check that the slope
    # is second-order.
    l = fit(log.(dts), log.(err_pm), 1)
    @test isapprox(l.coeffs[2], 2, atol=0.1)

    l = fit(log.(dts), log.(err_pd), 1)
    @test isapprox(l.coeffs[2], 2, atol=0.1)
end

function if1_integrand(se, obs, t)
    c0 = se.c0

    # Need to get the retarded time.
    R(τ) = [t - (τ[1] + norm(obs(t) .- se.y0dot(τ[1]))/c0)]
    result = nlsolve(R, [-0.1], autodiff=ADTypes.AutoForwardDiff())
    if !converged(result)
        @error "nlsolve retarded time calculation did not converge:\n$(result)"
    end
    τ = result.zero[1]

    # Position of source at the retarted time.
    y = se.y0dot(τ)

    # Position vector from source to observer.
    rv = obs(t) .- y

    # Distance from source to observer.
    r = AcousticAnalogies.norm_cs_safe(rv)

    # Unit vector pointing from source to observer.
    rhat = rv./r

    # First time derivative of rv.
    rv1dot = -se.y1dot(τ)

    # Mach number of the velocity of the source in the direction of the
    # observer.
    Mr = AcousticAnalogies.dot_cs_safe(-rv1dot/c0, rhat)

    # Now evaluate the integrand.
    vn = dot_cs_safe(se.y1dot(τ), se.nhat(τ))
    Q = se.ρ0*vn
    R11 = 1 / (r * (1 - Mr))
    A1 = R11
    p_m_integrand = Q*A1*se.dA/(4*pi)

    # Loading at the retarded time.
    f0dot = se.f0dot(τ)

    B1 = R11*rhat
    p_d_integrand_ff = dot_cs_safe(f0dot, B1)*se.dA/(4*pi*c0)
    R21 = R11/r
    C1 = c0*R21*rhat
    p_d_integrand_nf = dot_cs_safe(f0dot, C1)*se.dA/(4*pi*c0)
    return τ, p_m_integrand, p_d_integrand_ff, p_d_integrand_nf
end

@testset "Impermeable F1A tests" begin
    # rho = 1.226e6  # kg/m^3
    rho = 1.226e0  # kg/m^3
    c0 = 340.0  # m/s
    Rtip = 1.1684  # meters
    radii = 0.99932*Rtip
    dradii = (0.99932 - 0.99660)*Rtip  # m

    theta = 90.0*pi/180.0
    # theta = 10*pi/180
    x0 = [cos(theta), 0.0, sin(theta)].*100.0.*12.0.*0.0254  # 100 ft in meters
    # x0 = [cos(theta), 0.0, sin(theta)].*10.0.*12.0.*0.0254
    obs = StationaryAcousticObserver(x0)

    # Need the position and velocity of the source as a function of
    # source/retarded time. How do I want it to move? I want it to rotate around
    # an axis on the origin, pointing in the x direction.
    rpm = 2200
    omega = 2*pi/60*rpm
    period = 60/rpm
    fn = 180.66763939805125
    fc = 19.358679206883078
    # For the impermiable case, we just need the position through acceleration, and the loading and its time derivative.
    # y0dot(τ) = [0,  radii*cos(omega*τ), radii*sin(omega*τ)]
    # y1dot(τ) = [0, -omega*radii*sin(omega*τ), omega*radii*cos(omega*τ)]
    # y2dot(τ) = [0, -omega^2*radii*cos(omega*τ), -omega^2*radii*sin(omega*τ)]
    y0dot(τ) = [0.1*radii*sin(omega*τ),  radii*cos(omega*τ), radii*sin(omega*τ)]
    y1dot(τ) = [0.1*radii*omega*cos(omega*τ), -omega*radii*sin(omega*τ), omega*radii*cos(omega*τ)]
    y2dot(τ) = [-0.1*radii*omega^2*sin(omega*τ), -omega^2*radii*cos(omega*τ), -omega^2*radii*sin(omega*τ)]
    # f0dot(τ) = [-fn, -sin(omega*τ)*fc, cos(omega*τ)*fc]
    # f1dot(τ) = [0, -omega*cos(omega*τ)*fc, -omega*sin(omega*τ)*fc]
    f0dot(τ) = [-fn*cos(omega*τ), -sin(omega*τ)*fc, cos(omega*τ)*fc]
    f1dot(τ) = [fn*omega*sin(omega*τ), -omega*cos(omega*τ)*fc, -omega*sin(omega*τ)*fc]
    # But we also need the unit normal, etc..
    # What should the unit normal be?
    # Doesn't really need to be realistic.
    # But don't make it perpendicular to the velocity like I initially did---no monopole noise then!
    # Also, I'm not sure if I can arbitrarily mess with the normal vector and not do something with the position, velocity, etc.. of the source element and expect F1A to be consistent with F1.
    # Need to think about that.
    # But at least this case works.
    # No, that's all completely wrong---nhat and the position, etc are all independent.
    # I was forgetting about the quotient rule when writing down `nhatdot`, lol.
    # nhat(τ) = [1.0, -sin(omega*τ), cos(omega*τ)]./sqrt(1 + 1)
    # nhatdot(τ) = [0.0, -omega*cos(omega*τ), -omega*sin(omega*τ)]./sqrt(1 + 1)
    nhat(τ) = [τ^2, -sin(omega*τ), cos(omega*τ)]./sqrt(τ^2 + 1)
    # nhatdot(τ) = [2*τ, -omega*cos(omega*τ), -omega*sin(omega*τ)]./sqrt(τ^2 + 1)
    nhatdot(τ) = ([2*τ, -omega*cos(omega*τ), -omega*sin(omega*τ)].*sqrt(τ^2 + 1) - [τ^2, -sin(omega*τ), cos(omega*τ)].*(0.5*(τ^2 + 1)^(-0.5)*(2*τ)))./(τ^2 + 1)
    # I guess we need an area too.
    dA = 0.2
    t = 0.0 + 0.65*period
    sef1 = ImpermeableNonCompactF1ASourceElement(rho, c0, y0dot, y1dot, nothing, dA, nhat, nothing, f0dot, nothing, t)

    dt = period*0.5^4

    τ0, pmi0, pdiff0, pdinf0 = if1_integrand(sef1, obs, t)
    sef1a = ImpermeableNonCompactF1ASourceElement(rho, c0, y0dot(τ0), y1dot(τ0), y2dot(τ0), dA, nhat(τ0), nhatdot(τ0), f0dot(τ0), f1dot(τ0), τ0)
    apth = noise(sef1a, obs)

    err_prev_pm = nothing
    err_prev_pd = nothing
    dt_prev = nothing
    dt_curr = dt
    first_time = true

    err_pm = Vector{Float64}()
    err_pd = Vector{Float64}()
    dts = Vector{Float64}()
    ooa_pm = Vector{Float64}()
    ooa_pd = Vector{Float64}()
    for n in 1:7
        τ_1, pmi_1, pdiff_1, pdinf_1 = if1_integrand(sef1, obs, t-dt_curr)
        τ1, pmi1, pdiff1, pdinf1 = if1_integrand(sef1, obs, t+dt_curr)

        p_m_f1 = (pmi1 - pmi_1)/(2*dt_curr)
        p_d_f1 = (pdiff1 - pdiff_1)/(2*dt_curr) + pdinf0
        # println("n = $n, p_m_f1 = $p_m_f1, p_d_f1 = $p_d_f1")

        err_curr_pm = abs(p_m_f1 - apth.p_m)
        err_curr_pd = abs(p_d_f1 - apth.p_d)

        if first_time
            first_time = false
        else
            push!(ooa_pm, log(err_curr_pm/err_prev_pm)/log(dt_curr/dt_prev))
            push!(ooa_pd, log(err_curr_pd/err_prev_pd)/log(dt_curr/dt_prev))
        end

        push!(dts, dt_curr)
        push!(err_pm, err_curr_pm)
        push!(err_pd, err_curr_pd)

        dt_prev = dt_curr
        err_prev_pm = err_curr_pm
        err_prev_pd = err_curr_pd
        dt_curr = 0.5*dt_curr
    end

    # Fit a line through the errors on a log-log plot, then check that the slope
    # is second-order.
    # @show dts err_pm err_pd
    l = fit(log.(dts), log.(err_pm), 1)
    # println("monopole convergence rate = $(l.coeffs[2])")
    @test isapprox(l.coeffs[2], 2, atol=0.1)

    l = fit(log.(dts), log.(err_pd), 1)
    # println("dipole convergence rate = $(l.coeffs[2])")
    @test isapprox(l.coeffs[2], 2, atol=0.1)
end

end # module
