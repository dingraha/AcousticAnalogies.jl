module BrooksPopeMarcoliniTests

using AcousticAnalogies: AcousticAnalogies
using AcousticMetrics: AcousticMetrics
using DelimitedFiles: DelimitedFiles
using FLOWMath: linear
using Test

@testset "boundary layer thickness" begin
    @testset "zero angle of attack" begin
        @testset "untripped" begin
            # Get the digitized data from the BPM report plot.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure06-bl_thickness-untripped.csv")
            bpm_untripped = DelimitedFiles.readdlm(fname, ',')
            Re_c_1e6 = bpm_untripped[:, 1]
            deltastar0_c = bpm_untripped[:, 2]

            # Get the AcousticAnalogies.jl implementation.
            Re_c_1e6_jl = range(minimum(Re_c_1e6), maximum(Re_c_1e6); length=50)
            deltastar0_c_jl = AcousticAnalogies.bl_thickness_0.(Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()), Re_c_1e6_jl.*1e6)
            # Interpolate the BPM report data onto the uniform Re spacing.
            deltastar0_c_interp = linear(Re_c_1e6, deltastar0_c, Re_c_1e6_jl)

            # Find the scaled error.
            vmin, vmax = extrema(deltastar0_c)
            err = abs.(deltastar0_c_jl .- deltastar0_c_interp)/(vmax - vmin)
            @test maximum(err) < 0.041
        end
    end

    @testset "non-zero angle of attack, untripped" begin
        @testset "pressure side" begin
        end
    end
end

@testset "displacement thickness" begin
    @testset "zero angle of attack" begin
        @testset "tripped" begin
            # Get the digitized data from the BPM report plot.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure06-disp_thickness-tripped.csv")
            bpm_tripped = DelimitedFiles.readdlm(fname, ',')
            Re_c_1e6 = bpm_tripped[:, 1]
            deltastar0_c = bpm_tripped[:, 2]

            # Get the AcousticAnalogies.jl implementation.
            Re_c_1e6_jl = range(minimum(Re_c_1e6), maximum(Re_c_1e6); length=50)
            deltastar0_c_jl = AcousticAnalogies.disp_thickness_0.(Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()), Re_c_1e6_jl.*1e6)

            # Interpolate the BPM report data onto the uniform Re spacing.
            deltastar0_c_interp = linear(Re_c_1e6, deltastar0_c, Re_c_1e6_jl)

            # Find the scaled error.
            vmin, vmax = extrema(deltastar0_c)
            err = abs.(deltastar0_c_jl .- deltastar0_c_interp)/(vmax - vmin)
            @test maximum(err) < 0.05
        end

        @testset "untripped" begin
            # Get the digitized data from the BPM report plot.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure06-disp_thickness-untripped.csv")
            bpm_untripped = DelimitedFiles.readdlm(fname, ',')
            Re_c_1e6 = bpm_untripped[:, 1]
            deltastar0_c = bpm_untripped[:, 2]

            # Get the AcousticAnalogies.jl implementation.
            Re_c_1e6_jl = range(minimum(Re_c_1e6), maximum(Re_c_1e6); length=50)
            deltastar0_c_jl = AcousticAnalogies.disp_thickness_0.(Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()), Re_c_1e6_jl.*1e6)
            # Interpolate the BPM report data onto the uniform Re spacing.
            deltastar0_c_interp = linear(Re_c_1e6, deltastar0_c, Re_c_1e6_jl)

            # Find the scaled error.
            vmin, vmax = extrema(deltastar0_c)
            err = abs.(deltastar0_c_jl .- deltastar0_c_interp)/(vmax - vmin)
            @test maximum(err) < 0.02
        end
    end

    @testset "non-zero angle of attack, tripped" begin
        @testset "pressure side" begin
            # Get the digitized data from the BPM report plot.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure07-pressure_side.csv")
            bpm_pressure_side = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm_pressure_side[:, 1]
            deltastar_bpm = bpm_pressure_side[:, 2]

            # Get the AcousticAnalogies.jl implementation.
            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=50)
            deltastar_jl = AcousticAnalogies._disp_thickness_p.(Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()), alpha_deg_jl.*pi/180)

            # Interpolate the BPM report data onto the uniform alpha spacing.
            deltastar_bpm_interp = linear(alpha_deg, deltastar_bpm, alpha_deg_jl)

            # Find the scaled error.
            vmin, vmax = extrema(deltastar_bpm)
            err = abs.(deltastar_jl .- deltastar_bpm_interp)./(vmax - vmin)
            @test maximum(err) < 0.06
        end

        @testset "suction side" begin
            # Get the digitized data from the BPM report plot.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure07-suction_side.csv")
            bpm_suction_side = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm_suction_side[:, 1]
            deltastar_bpm = bpm_suction_side[:, 2]

            # Get the AcousticAnalogies.jl implementation.
            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=50)
            deltastar_jl = AcousticAnalogies._disp_thickness_s.(Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()), alpha_deg_jl.*pi/180)

            # Interpolate the BPM report data onto the uniform alpha spacing.
            deltastar_bpm_interp = linear(alpha_deg, deltastar_bpm, alpha_deg_jl)

            # Find the scaled error.
            vmin, vmax = extrema(deltastar_bpm)
            err = abs.(deltastar_jl .- deltastar_bpm_interp)./(vmax - vmin)
            @test maximum(err) < 0.04
        end
    end

    @testset "non-zero angle of attack, untripped" begin
        @testset "boundary layer thickness, pressure side" begin
            # Get the digitized data from the BPM report plot.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure08-bl_thickness-pressure_side.csv")
            bpm_pressure_side = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm_pressure_side[:, 1]
            delta_bpm = bpm_pressure_side[:, 2]

            # Get the AcousticAnalogies.jl implementation.
            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=50)
            delta_jl = AcousticAnalogies.bl_thickness_p.(Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()), alpha_deg_jl.*pi/180)

            # Interpolate the BPM report onto the uniform alpha spacing.
            delta_bpm_interp = linear(alpha_deg, delta_bpm, alpha_deg_jl)

            # Find the scaled error.
            vmin, vmax = extrema(delta_bpm)
            err = abs.(delta_jl .- delta_bpm_interp)./(vmax - vmin)
            @test maximum(err) < 0.037
        end
        @testset "displacement thickness, pressure side" begin
            # Get the digitized data from the BPM report plot.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure08-pressure_side.csv")
            bpm_pressure_side = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm_pressure_side[:, 1]
            deltastar_bpm = bpm_pressure_side[:, 2]

            # Get the AcousticAnalogies.jl implementation.
            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=50)
            deltastar_jl = AcousticAnalogies._disp_thickness_p.(Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()), alpha_deg_jl.*pi/180)

            # Interpolate the BPM report onto the uniform alpha spacing.
            deltastar_bpm_interp = linear(alpha_deg, deltastar_bpm, alpha_deg_jl)

            # Find the scaled error.
            vmin, vmax = extrema(deltastar_bpm)
            err = abs.(deltastar_jl .- deltastar_bpm_interp)./(vmax - vmin)
            # This is dumb. Maybe I have a bug?
            @test maximum(err) < 0.11
        end

        @testset "displacement thinckness, suction side" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure08-suction_side.csv")
            bpm_suction_side = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm_suction_side[:, 1]
            deltastar_bpm = bpm_suction_side[:, 2]

            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=50)
            deltastar_jl = AcousticAnalogies._disp_thickness_s.(Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()), alpha_deg_jl.*pi/180)

            deltastar_bpm_interp = linear(alpha_deg, deltastar_bpm, alpha_deg_jl)

            vmin, vmax = extrema(deltastar_bpm)
            err = abs.(deltastar_jl .- deltastar_bpm_interp)./(vmax - vmin)
            # This is dumb. Maybe I have a bug?
            @test maximum(err) < 0.081
        end
    end
end

@testset "shape functions" begin
    @testset "TBL-TE" begin
        @testset "St_1" begin
            @test isapprox(AcousticAnalogies.St_1(0.093), 0.081; atol=0.0022)
            @test isapprox(AcousticAnalogies.St_1(0.116), 0.071; atol=0.002)
            @test isapprox(AcousticAnalogies.St_1(0.163), 0.059; atol=0.0004)
            @test isapprox(AcousticAnalogies.St_1(0.209), 0.051; atol=0.0002)
        end

        @testset "K_1" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure77.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            Re_c_bpm = bpm[:, 1]
            K_1_bpm = bpm[:, 2]

            Re_c_jl = range(minimum(Re_c_bpm), maximum(Re_c_bpm); length=50)
            K_1_jl = AcousticAnalogies.K_1.(Re_c_jl)

            K_1_interp = linear(Re_c_bpm, K_1_bpm, Re_c_jl)

            vmin, vmax = extrema(K_1_bpm)
            err = abs.(K_1_jl .- K_1_interp)./(vmax - vmin)
            @test maximum(err) < 0.012
        end

        @testset "A" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure78-A_min.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_St_peak_bpm = bpm[:, 1]
            A = bpm[:, 2]

            St_St_peak_jl = range(minimum(St_St_peak_bpm), maximum(St_St_peak_bpm); length=50)
            A_jl = AcousticAnalogies.A.(St_St_peak_jl, 9.5e4)

            # Interpolate:
            A_interp = linear(St_St_peak_bpm, A, St_St_peak_jl)
            
            # Check error.
            vmin, vmax = extrema(A)
            err = abs.(A_jl .- A_interp)./(vmax - vmin)
            @test maximum(err) < 0.057

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure78-A_max.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_St_peak_bpm = bpm[:, 1]
            A = bpm[:, 2]

            St_St_peak_jl = range(minimum(St_St_peak_bpm), maximum(St_St_peak_bpm); length=50)
            A_jl = AcousticAnalogies.A.(St_St_peak_jl, 8.58e5)

            # Interpolate:
            A_interp = linear(St_St_peak_bpm, A, St_St_peak_jl)
            
            # Check error.
            vmin, vmax = extrema(A)
            err = abs.(A_jl .- A_interp)./(vmax - vmin)
            @test maximum(err) < 0.021
        end

        @testset "B" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure78-B_min.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_St_peak_bpm = bpm[:, 1]
            B = bpm[:, 2]

            St_St_peak_jl = range(minimum(St_St_peak_bpm), maximum(St_St_peak_bpm); length=50)
            B_jl = AcousticAnalogies.B.(St_St_peak_jl, 9.5e4)

            # Interpolate:
            B_interp = linear(St_St_peak_bpm, B, St_St_peak_jl)
            
            # Check error.
            vmin, vmax = extrema(B)
            err = abs.(B_jl .- B_interp)./(vmax - vmin)
            @test maximum(err) < 0.057

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure78-B_max.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_St_peak_bpm = bpm[:, 1]
            B = bpm[:, 2]

            St_St_peak_jl = range(minimum(St_St_peak_bpm), maximum(St_St_peak_bpm); length=50)
            B_jl = AcousticAnalogies.B.(St_St_peak_jl, 8.58e5)

            # Interpolate:
            B_interp = linear(St_St_peak_bpm, B, St_St_peak_jl)
            
            # Check error.
            vmin, vmax = extrema(B)
            err = abs.(B_jl .- B_interp)./(vmax - vmin)
            @test maximum(err) < 0.020
        end

        @testset "St_2" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure80-M0.093.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm[:, 1]
            St_2 = bpm[:, 2]

            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=50)
            St_2_jl = AcousticAnalogies.St_2.(AcousticAnalogies.St_1(0.093), alpha_deg_jl.*pi/180)

            # Interpolate:
            St_2_interp = linear(alpha_deg, St_2, alpha_deg_jl)

            # Check error.
            vmin, vmax = extrema(St_2)
            err = abs.(St_2_jl .- St_2_interp)./(vmax - vmin)
            @test maximum(err) < 0.023

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure80-M0.209.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm[:, 1]
            St_2 = bpm[:, 2]

            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=50)
            St_2_jl = AcousticAnalogies.St_2.(AcousticAnalogies.St_1(0.209), alpha_deg_jl.*pi/180)

            # Interpolate:
            St_2_interp = linear(alpha_deg, St_2, alpha_deg_jl)

            # Check error.
            vmin, vmax = extrema(St_2)
            err = abs.(St_2_jl .- St_2_interp)./(vmax - vmin)
            @test maximum(err) < 0.011

        end

        @testset "K_2" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure82-M0.093.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm[:, 1]
            K_2_K_1 = bpm[:, 2]

            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=200)
            K_2_K_1_jl = AcousticAnalogies.K_2.(1e6, 0.093, alpha_deg_jl.*pi/180) .- AcousticAnalogies.K_1(1e6)

            # Interpolate:
            K_2_K_1_interp = linear(alpha_deg, K_2_K_1, alpha_deg_jl)

            # Check error.
            vmin, vmax = extrema(K_2_K_1)
            err = abs.(K_2_K_1_jl .- K_2_K_1_interp)./(vmax - vmin)
            # The curve is almost vertical at low angles of attack, so a small error in the digitization results in big differences.
            @test maximum(err[2:end]) < 0.027

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure82-M0.116.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm[:, 1]
            K_2_K_1 = bpm[:, 2]

            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=200)
            K_2_K_1_jl = AcousticAnalogies.K_2.(1e6, 0.116, alpha_deg_jl.*pi/180) .- AcousticAnalogies.K_1(1e6)

            # Interpolate:
            K_2_K_1_interp = linear(alpha_deg, K_2_K_1, alpha_deg_jl)

            # Check error.
            vmin, vmax = extrema(K_2_K_1)
            err = abs.(K_2_K_1_jl .- K_2_K_1_interp)./(vmax - vmin)
            # There's a branch for low angles of attack that sets K_2 - K_1 to
            # -1000, but I can't see that in the digitized plots, so the first
            # point is bad.
            @test K_2_K_1_jl[1] ≈ -1000
            @test maximum(err[2:end]) < 0.022

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure82-M0.163.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm[:, 1]
            K_2_K_1 = bpm[:, 2]

            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=200)
            K_2_K_1_jl = AcousticAnalogies.K_2.(1e6, 0.163, alpha_deg_jl.*pi/180) .- AcousticAnalogies.K_1(1e6)

            # Interpolate:
            K_2_K_1_interp = linear(alpha_deg, K_2_K_1, alpha_deg_jl)

            # Check error.
            vmin, vmax = extrema(K_2_K_1)
            err = abs.(K_2_K_1_jl .- K_2_K_1_interp)./(vmax - vmin)
            # There's a branch for low angles of attack that sets K_2 - K_1 to
            # -1000, but I can't see that in the digitized plots, so the first
            # point is bad.
            @test K_2_K_1_jl[1] ≈ -1000.0
            @test maximum(err[2:end]) < 0.020

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure82-M0.209.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            alpha_deg = bpm[:, 1]
            K_2_K_1 = bpm[:, 2]

            alpha_deg_jl = range(minimum(alpha_deg), maximum(alpha_deg); length=200)
            K_2_K_1_jl = AcousticAnalogies.K_2.(1e6, 0.209, alpha_deg_jl.*pi/180) .- AcousticAnalogies.K_1(1e6)

            # Interpolate:
            K_2_K_1_interp = linear(alpha_deg, K_2_K_1, alpha_deg_jl)
            
            # Check error.
            vmin, vmax = extrema(K_2_K_1)
            err = abs.(K_2_K_1_jl .- K_2_K_1_interp)./(vmax - vmin)
            @test maximum(err) < 0.024
        end

        @testset "BPM Figure 11a" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure11-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1] # This is in kHz.
            SPL_s = bpm[:, 2]

            # At zero angle of attack the pressure and suction side predictions are the same.
            f_p = f_s
            SPL_p = SPL_s

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 30.48e-2  # chord in meters
            U = 71.3  # freestream velocity in m/s
            M = 0.209  # Mach number, corresponds to U = 71.3 m/s in BPM report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))
            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.029

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.029

        end

        @testset "BPM Figure 11d" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure11-d-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # At zero angle of attack the pressure and suction side predictions are the same.
            f_p = f_s
            SPL_p = SPL_s

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 30.48e-2  # chord in meters
            U = 31.7  # freestream velocity in m/s
            M = 0.093  # Mach number, corresponds to U = 31.7 m/s in BPM report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))
            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.015

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.015

        end

        @testset "BPM Figure 12a" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure12-U71.3-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure12-U71.3-TBL-TE-pressure.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure12-U71.3-separation.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_alpha = bpm[:, 1]
            SPL_alpha = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 30.48e-2  # chord in meters
            U = 71.3  # freestream velocity in m/s
            M = 0.209  # Mach number, corresponds to U = 71.3 m/s in BPM report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 1.5*pi/180

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            println("BPM Figure 12a alphastar = $(alphastar) rad, $(alphastar*pi/180) deg")
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))
            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.022

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            @test maximum(err) < 0.017

            SPL_alpha_jl_interp = linear(f_jl, SPL_alpha_jl, f_alpha.*1e3)
            vmin, vmax = extrema(SPL_alpha)
            err = abs.(SPL_alpha_jl_interp .- SPL_alpha)./(vmax - vmin)
            @test maximum(err) < 0.037
        end

        @testset "BPM Figure 26a" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure26-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # # Pressure and suction sides are the same for zero angle of attack.
            # f_p = f_s
            # SPL_p = SPL_s

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 10.16e-2  # chord in meters
            U = 71.3  # freestream velocity in m/s
            M = 0.209  # Mach number, corresponds to U = 71.3 m/s in BPM report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.015
        end

        @testset "BPM Figure 26d" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure26-d-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # # Pressure and suction sides are the same for zero angle of attack.
            # f_p = f_s
            # SPL_p = SPL_s

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 10.16e-2  # chord in meters
            U = 31.7  # freestream velocity in m/s
            M = 0.093  # Mach number, corresponds to U = 31.7 m/s in BPM report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.032

        end

        @testset "BPM Figure 28a" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure28-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure28-a-TBL-TE-pressure.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure28-a-separation.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_alpha = bpm[:, 1]
            SPL_alpha = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 10.16e-2  # chord in meters
            U = 71.3  # freestream velocity in m/s
            M = 0.209  # Mach number, corresponds to U = 71.3 m/s in BPM report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 6.7*pi/180

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.036

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            @test maximum(err) < 0.075

            SPL_alpha_jl_interp = linear(f_jl, SPL_alpha_jl, f_alpha.*1e3)
            vmin, vmax = extrema(SPL_alpha)
            err = abs.(SPL_alpha_jl_interp .- SPL_alpha)./(vmax - vmin)
            @test maximum(err) < 0.039
        end


        @testset "BPM Figure 28d" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure28-d-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure28-d-TBL-TE-pressure.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure28-d-separation.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_alpha = bpm[:, 1]
            SPL_alpha = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 10.16e-2  # chord in meters
            U = 31.7  # freestream velocity in m/s
            M = 0.093  # mach number, corresponds to u = 31.7 m/s in bpm report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 6.7*pi/180
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            # @show maximum(err)
            @test maximum(err) < 0.021

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            # @show maximum(err)
            @test maximum(err) < 0.042

            SPL_alpha_jl_interp = linear(f_jl, SPL_alpha_jl, f_alpha.*1e3)
            vmin, vmax = extrema(SPL_alpha)
            err = abs.(SPL_alpha_jl_interp .- SPL_alpha)./(vmax - vmin)
            # @show maximum(err)
            @test maximum(err) < 0.040
        end

        @testset "BPM Figure 38d" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure38-d-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # Pressure and suction sides are the same for zero angle of attack.
            f_p = f_s
            SPL_p = SPL_s

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 2.54e-2  # chord in meters
            U = 31.7  # freestream velocity in m/s
            M = 0.093  # mach number, corresponds to u = 31.7 m/s in bpm report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            # @show maximum(err)
            @test maximum(err) < 0.026

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            # @show maximum(err)
            @test maximum(err) < 0.026

            # SPL_alpha_jl_interp = linear(f_jl, SPL_alpha_jl, f_alpha.*1e3)
            # vmin, vmax = extrema(SPL_alpha)
            # err = abs.(SPL_alpha_jl_interp .- SPL_alpha)./(vmax - vmin)
            # @show maximum(err)
            # # @test maximum(err) < 0.040
        end

        @testset "BPM Figure 39d" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure39-d-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure39-d-TBL-TE-pressure.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure39-d-separation.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_alpha = bpm[:, 1]
            SPL_alpha = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 2.54e-2  # chord in meters
            U = 31.7  # freestream velocity in m/s
            M = 0.093  # mach number, corresponds to u = 31.7 m/s in bpm report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 4.8*pi/180
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            # @show maximum(err)
            @test maximum(err) < 0.036

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            # @show maximum(err)
            @test maximum(err) < 0.043

            SPL_alpha_jl_interp = linear(f_jl, SPL_alpha_jl, f_alpha.*1e3)
            vmin, vmax = extrema(SPL_alpha)
            err = abs.(SPL_alpha_jl_interp .- SPL_alpha)./(vmax - vmin)
            # @show maximum(err)
            @test maximum(err) < 0.039
        end

        @testset "BPM Figure 45a" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure45-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure45-a-TBL-TE-pressure.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure45-a-separation.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_alpha = bpm[:, 1]
            SPL_alpha = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure45-a-LBL-VS.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_lbl_vs = bpm[:, 1]
            SPL_lbl_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 30.48e-2  # chord in meters
            U = 71.3  # freestream velocity in m/s
            M = 0.209  # Mach number, corresponds to U = 71.3 m/s in BPM report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 1.5*pi/180

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)

            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            SPL_lbl_vs_jl = AcousticAnalogies.LBL_VS.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))

            # The agreement with these ones aren't so great.
            # Might be better if I grabbed the listing in the BPM appendix?
            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.037

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            @test maximum(err) < 0.058

            SPL_alpha_jl_interp = linear(f_jl, SPL_alpha_jl, f_alpha.*1e3)
            vmin, vmax = extrema(SPL_alpha)
            err = abs.(SPL_alpha_jl_interp .- SPL_alpha)./(vmax - vmin)
            @test maximum(err) < 0.091

            SPL_lbl_vs_jl_interp = linear(f_jl, SPL_lbl_vs_jl, f_lbl_vs.*1e3)
            vmin, vmax = extrema(SPL_lbl_vs)
            err = abs.(SPL_lbl_vs_jl_interp .- SPL_lbl_vs)./(vmax - vmin)
            @test maximum(err) < 0.053
        end

        @testset "BPM Figure 69a" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure69-a-separation.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_alpha = bpm[:, 1]
            SPL_alpha = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 5.08e-2  # chord in meters
            U = 71.3  # freestream velocity in m/s
            M = 0.209  # Mach number, corresponds to U = 71.3 m/s in BPM report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 15.4*pi/180
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 3)

            @test all(SPL_s_jl .≈ -100)
            @test all(SPL_p_jl .≈ -100)

            SPL_alpha_jl_interp = linear(f_jl, SPL_alpha_jl, f_alpha.*1e3)
            vmin, vmax = extrema(SPL_alpha)
            err = abs.(SPL_alpha_jl_interp .- SPL_alpha)./(vmax - vmin)
            @test maximum(err) < 0.033
        end
    end

    @testset "LBL-VS" begin
        @testset "St_1_prime" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure86-St_1_prime.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            Re_c_bpm = bpm[:, 1]
            St_1_prime_bpm = bpm[:, 2]

            Re_c_jl = 10.0.^(range(4, 7; length=100))
            St_1_prime_jl = AcousticAnalogies.St_1_prime.(Re_c_jl)

            St_1_prime_interp = linear(Re_c_bpm, St_1_prime_bpm, Re_c_jl)
            vmin, vmax = extrema(St_1_prime_bpm)
            err = abs.(St_1_prime_interp .- St_1_prime_jl)./(vmax - vmin)
            @test maximum(err) < 0.04
        end

        @testset "G1" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure85-G1.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            e_bpm = bpm[:, 1]
            G1_bpm = bpm[:, 2]

            e_jl = 10.0.^(range(-1, 1; length=101))
            G1_jl = AcousticAnalogies.G1.(e_jl)

            G1_interp = linear(e_jl, G1_jl, e_bpm)
            vmin, vmax = extrema(G1_bpm)
            err = abs.(G1_interp .- G1_bpm)./(vmax - vmin)
            @test maximum(err) < 0.033
        end

        @testset "St_peak_prime_alphastar" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure87.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            alphastar_bpm = bpm[:, 1]
            St_peak_ratio_bpm = bpm[:, 2]

            St_1_prime = 0.25  # Just make up a value, since we're multiplying and then dividing by it anyway.
            alphastar_jl = range(0.0*pi/180, 7.0*pi/180; length=21)
            St_peak_ratio_jl = AcousticAnalogies.St_peak_prime.(St_1_prime, alphastar_jl)./St_1_prime

            St_peak_ratio_interp = linear(alphastar_jl.*180/pi, St_peak_ratio_jl, alphastar_bpm)
            vmin, vmax = extrema(St_peak_ratio_bpm)
            err = abs.(St_peak_ratio_interp .- St_peak_ratio_bpm)./(vmax - vmin)
            @test maximum(err) < 0.031
        end

        @testset "G2" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure89.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            Re_ratio_bpm = bpm[:, 1]
            G2_bpm = bpm[:, 2]

            Re_ratio_jl = 10.0.^range(-1, 1, length=51)
            G2_jl = AcousticAnalogies.G2.(Re_ratio_jl)

            G2_interp = linear(Re_ratio_jl, G2_jl, Re_ratio_bpm)
            vmin, vmax = extrema(G2_interp)
            err = abs.(G2_interp .- G2_bpm)./(vmax - vmin)
            @test maximum(err) < 0.024
        end

        @testset "G2 + G3" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure88-G2-alpha0.csv")
            alphastar = 0.0*pi/180
            bpm = DelimitedFiles.readdlm(fname, ',')
            Re_c_bpm = bpm[:, 1]
            G2_bpm = bpm[:, 2]

            Re_c_jl = 10.0.^range(log10(first(Re_c_bpm)), log10(last(Re_c_bpm)), length=51)
            Re_c0 = AcousticAnalogies.Re_c0(alphastar)
            Re_ratio_jl = Re_c_jl./Re_c0
            G2_jl = AcousticAnalogies.G2.(Re_ratio_jl) .+ AcousticAnalogies.G3.(alphastar)

            G2_interp = linear(Re_c_jl, G2_jl, Re_c_bpm)
            vmin, vmax = extrema(G2_interp)
            err = abs.(G2_interp .- G2_bpm)./(vmax - vmin)
            @test maximum(err) < 0.013

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure88-G2-alpha6.csv")
            alphastar = 6.0*pi/180
            bpm = DelimitedFiles.readdlm(fname, ',')
            Re_c_bpm = bpm[:, 1]
            G2_bpm = bpm[:, 2]

            Re_c_jl = 10.0.^range(log10(first(Re_c_bpm)), log10(last(Re_c_bpm)), length=51)
            Re_c0 = AcousticAnalogies.Re_c0(alphastar)
            Re_ratio_jl = Re_c_jl./Re_c0
            G2_jl = AcousticAnalogies.G2.(Re_ratio_jl) .+ AcousticAnalogies.G3.(alphastar)

            G2_interp = linear(Re_c_jl, G2_jl, Re_c_bpm)
            vmin, vmax = extrema(G2_interp)
            err = abs.(G2_interp .- G2_bpm)./(vmax - vmin)
            @test maximum(err) < 0.030
        end

        @testset "BPM Figure 54a" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure54-a-LBL-VS.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_lbl_vs = bpm[:, 1]
            SPL_lbl_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 15.24e-2  # chord in meters
            U = 71.3  # freestream velocity in m/s
            M = 0.209  # Mach number, corresponds to U = 71.3 m/s in BPM report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 2.7*pi/180
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_lbl_vs_jl = AcousticAnalogies.LBL_VS.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))

            SPL_lbl_vs_jl_interp = linear(f_jl, SPL_lbl_vs_jl, f_lbl_vs.*1e3)
            vmin, vmax = extrema(SPL_lbl_vs)
            err = abs.(SPL_lbl_vs_jl_interp .- SPL_lbl_vs)./(vmax - vmin)
            @test maximum(err) < 0.026
        end

        @testset "BPM Figure 60d" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure60-d-LBL-VS.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_lbl_vs = bpm[:, 1]
            SPL_lbl_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 10.16e-2  # chord in meters
            U = 31.7  # freestream velocity in m/s
            M = 0.093  # mach number, corresponds to u = 31.7 m/s in bpm report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 3.3*pi/180
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_lbl_vs_jl = AcousticAnalogies.LBL_VS.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))

            SPL_lbl_vs_jl_interp = linear(f_jl, SPL_lbl_vs_jl, f_lbl_vs.*1e3)
            vmin, vmax = extrema(SPL_lbl_vs)
            err = abs.(SPL_lbl_vs_jl_interp .- SPL_lbl_vs)./(vmax - vmin)
            @test maximum(err) < 0.026
        end

        @testset "BPM Figure 65d" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure65-d-LBL-VS.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_lbl_vs = bpm[:, 1]
            SPL_lbl_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 5.08e-2  # chord in meters
            U = 31.7  # freestream velocity in m/s
            M = 0.093  # mach number, corresponds to u = 31.7 m/s in bpm report
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_lbl_vs_jl = AcousticAnalogies.LBL_VS.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))

            SPL_lbl_vs_jl_interp = linear(f_jl, SPL_lbl_vs_jl, f_lbl_vs.*1e3)
            vmin, vmax = extrema(SPL_lbl_vs)
            err = abs.(SPL_lbl_vs_jl_interp .- SPL_lbl_vs)./(vmax - vmin)
            @test maximum(err) < 0.021
        end

    end

    @testset "Tip Vortex" begin

        @testset "BPM Figure 91" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure91-tip.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_tip = bpm[:, 1]
            SPL_tip = bpm[:, 2]

            # L = 30.48e-2  # span in meters
            chord = 15.24e-2  # chord in meters
            speedofsound = 340.46
            U = 71.3  # freestream velocity in m/s
            # M = 0.209  # Mach number, corresponds to U = 71.3 m/s in BPM report
            M = U/speedofsound
            M_c = 0.8*M
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            alphatip = 0.71*10.8*pi/180
            # Equation 64 in the BPM report.
            M_max = (1 + 0.036*(alphatip*180/pi))*M
            U_max = M_max*speedofsound
            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_tip_jl = AcousticAnalogies.TIP.(f_jl, chord, M, M_c, U_max, M_max, r_e, θ_e, Φ_e, alphatip, Ref(AcousticAnalogies.RoundedTip()))

            SPL_tip_jl_interp = linear(f_jl, SPL_tip_jl, f_tip.*1e3)
            vmin, vmax = extrema(SPL_tip)
            err = abs.(SPL_tip_jl_interp .- SPL_tip)./(vmax - vmin)
            @test maximum(err) < 0.047
        end

    end

    @testset "TEB-VS" begin
        @testset "St_3prime_peak" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure95-0Psi.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            h_over_deltastar_0Psi = bpm[:, 1]
            St_3prime_peak_0Psi = bpm[:, 2]
             
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure95-14Psi.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            h_over_deltastar_14Psi = bpm[:, 1]
            St_3prime_peak_14Psi = bpm[:, 2]

            h_over_deltastar_jl = 10.0.^(range(-1, 1; length=101))
            St_3prime_peak_0Psi_jl = AcousticAnalogies.St_3prime_peak.(h_over_deltastar_jl, 0.0*pi/180)
            St_3prime_peak_14Psi_jl = AcousticAnalogies.St_3prime_peak.(h_over_deltastar_jl, 14.0*pi/180)

            St_3prime_peak_0Psi_interp = linear(h_over_deltastar_jl, St_3prime_peak_0Psi_jl, h_over_deltastar_0Psi)
            vmin, vmax = extrema(St_3prime_peak_0Psi)
            err = abs.(St_3prime_peak_0Psi_interp .- St_3prime_peak_0Psi)./(vmax - vmin)
            @test maximum(err) < 0.070

            St_3prime_peak_14Psi_interp = linear(h_over_deltastar_jl, St_3prime_peak_14Psi_jl, h_over_deltastar_14Psi)
            vmin, vmax = extrema(St_3prime_peak_14Psi)
            err = abs.(St_3prime_peak_14Psi_interp .- St_3prime_peak_14Psi)./(vmax - vmin)
            @test maximum(err) < 0.049
        end

        @testset "G4" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure96-0Psi.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            h_over_deltastar_0Psi = bpm[:, 1]
            G4_0Psi = bpm[:, 2]
             
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure96-14Psi.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            h_over_deltastar_14Psi = bpm[:, 1]
            G4_14Psi = bpm[:, 2]

            h_over_deltastar_jl = 10.0.^(range(-1, 1; length=51))
            G4_0Psi_jl = AcousticAnalogies.G4.(h_over_deltastar_jl, 0.0*pi/180)
            G4_14Psi_jl = AcousticAnalogies.G4.(h_over_deltastar_jl, 14.0*pi/180)

            G4_0Psi_interp = linear(h_over_deltastar_jl, G4_0Psi_jl, h_over_deltastar_0Psi)
            vmin, vmax = extrema(G4_0Psi)
            err = abs.(G4_0Psi_interp .- G4_0Psi)./(vmax - vmin)
            @test maximum(err) < 0.033

            G4_14Psi_interp = linear(h_over_deltastar_jl, G4_14Psi_jl, h_over_deltastar_14Psi)
            vmin, vmax = extrema(G4_14Psi)
            err = abs.(G4_14Psi_interp .- G4_14Psi)./(vmax - vmin)
            @test maximum(err) < 0.024

        end

        @testset "G5, Psi = 14°" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi14-h_over_deltastar0p25.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_0p25 = bpm[:, 1]
            G5_14Psi_h_over_deltastar_avg0p25 = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi14-h_over_deltastar0p43.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_0p43 = bpm[:, 1]
            G5_14Psi_h_over_deltastar_avg0p43 = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi14-h_over_deltastar0p50.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_0p50 = bpm[:, 1]
            G5_14Psi_h_over_deltastar_avg0p50 = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi14-h_over_deltastar0p54.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_0p54 = bpm[:, 1]
            G5_14Psi_h_over_deltastar_avg0p54 = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi14-h_over_deltastar0p62.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_0p62 = bpm[:, 1]
            G5_14Psi_h_over_deltastar_avg0p62 = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi14-h_over_deltastar1p20.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_1p20 = bpm[:, 1]
            G5_14Psi_h_over_deltastar_avg1p20 = bpm[:, 2]

            St_3prime_over_St_3prime_peak_jl = 10.0.^(range(-1, 10; length=1001))
            G5_14Psi_h_over_deltastar_avg0p25_jl = AcousticAnalogies.G5_Psi14.(0.25, St_3prime_over_St_3prime_peak_jl)
            G5_14Psi_h_over_deltastar_avg0p43_jl = AcousticAnalogies.G5_Psi14.(0.43, St_3prime_over_St_3prime_peak_jl)
            G5_14Psi_h_over_deltastar_avg0p50_jl = AcousticAnalogies.G5_Psi14.(0.50, St_3prime_over_St_3prime_peak_jl)
            G5_14Psi_h_over_deltastar_avg0p54_jl = AcousticAnalogies.G5_Psi14.(0.54, St_3prime_over_St_3prime_peak_jl)
            G5_14Psi_h_over_deltastar_avg0p62_jl = AcousticAnalogies.G5_Psi14.(0.62, St_3prime_over_St_3prime_peak_jl)
            G5_14Psi_h_over_deltastar_avg1p20_jl = AcousticAnalogies.G5_Psi14.(1.20, St_3prime_over_St_3prime_peak_jl)

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_14Psi_h_over_deltastar_avg0p25_jl, St_3prime_over_St_3prime_peak_0p25)
            vmin, vmax = extrema(G5_14Psi_h_over_deltastar_avg0p25)
            err = abs.(interp .- G5_14Psi_h_over_deltastar_avg0p25)/(vmax - vmin)
            @test maximum(err) < 0.074

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_14Psi_h_over_deltastar_avg0p43_jl, St_3prime_over_St_3prime_peak_0p43)
            vmin, vmax = extrema(G5_14Psi_h_over_deltastar_avg0p43)
            err = abs.(interp .- G5_14Psi_h_over_deltastar_avg0p43)/(vmax - vmin)
            @test maximum(err) < 0.072

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_14Psi_h_over_deltastar_avg0p50_jl, St_3prime_over_St_3prime_peak_0p50)
            vmin, vmax = extrema(G5_14Psi_h_over_deltastar_avg0p50)
            err = abs.(interp .- G5_14Psi_h_over_deltastar_avg0p50)/(vmax - vmin)
            @test maximum(err) < 0.072

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_14Psi_h_over_deltastar_avg0p54_jl, St_3prime_over_St_3prime_peak_0p54)
            vmin, vmax = extrema(G5_14Psi_h_over_deltastar_avg0p54)
            err = abs.(interp .- G5_14Psi_h_over_deltastar_avg0p54)/(vmax - vmin)
            @test maximum(err) < 0.074

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_14Psi_h_over_deltastar_avg0p62_jl, St_3prime_over_St_3prime_peak_0p62)
            vmin, vmax = extrema(G5_14Psi_h_over_deltastar_avg0p62)
            err = abs.(interp .- G5_14Psi_h_over_deltastar_avg0p62)/(vmax - vmin)
            @test maximum(err) < 0.073

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_14Psi_h_over_deltastar_avg1p20_jl, St_3prime_over_St_3prime_peak_1p20)
            vmin, vmax = extrema(G5_14Psi_h_over_deltastar_avg1p20)
            err = abs.(interp .- G5_14Psi_h_over_deltastar_avg1p20)/(vmax - vmin)
            # The lower end of this case is really bad.
            # Not sure why. :-(
            @test maximum(err[1:22]) < 0.31
            @test maximum(err[23:end]) < 0.087
        end

        @testset "G5, Psi = 0°" begin
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi0-h_over_deltastar0p25.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_0p25 = bpm[:, 1]
            G5_0Psi_h_over_deltastar_avg0p25 = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi0-h_over_deltastar0p43.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_0p43 = bpm[:, 1]
            G5_0Psi_h_over_deltastar_avg0p43 = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi0-h_over_deltastar0p50.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_0p50 = bpm[:, 1]
            G5_0Psi_h_over_deltastar_avg0p50 = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi0-h_over_deltastar0p54.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_0p54 = bpm[:, 1]
            G5_0Psi_h_over_deltastar_avg0p54 = bpm[:, 2]

            # fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi0-h_over_deltastar0p62.csv")
            # bpm = DelimitedFiles.readdlm(fname, ',')
            # St_3prime_over_St_3prime_peak_0p62 = bpm[:, 1]
            # G5_0Psi_h_over_deltastar_avg0p62 = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure97-Psi0-h_over_deltastar1p20.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            St_3prime_over_St_3prime_peak_1p20 = bpm[:, 1]
            G5_0Psi_h_over_deltastar_avg1p20 = bpm[:, 2]

            St_3prime_over_St_3prime_peak_jl = 10.0.^(range(-1, 10; length=1001))
            G5_0Psi_h_over_deltastar_avg0p25_jl = AcousticAnalogies.G5_Psi0.(0.25, St_3prime_over_St_3prime_peak_jl)
            G5_0Psi_h_over_deltastar_avg0p43_jl = AcousticAnalogies.G5_Psi0.(0.43, St_3prime_over_St_3prime_peak_jl)
            G5_0Psi_h_over_deltastar_avg0p50_jl = AcousticAnalogies.G5_Psi0.(0.50, St_3prime_over_St_3prime_peak_jl)
            G5_0Psi_h_over_deltastar_avg0p54_jl = AcousticAnalogies.G5_Psi0.(0.54, St_3prime_over_St_3prime_peak_jl)
            # G5_0Psi_h_over_deltastar_avg0p62_jl = AcousticAnalogies.G5_Psi0.(0.62, St_3prime_over_St_3prime_peak_jl)
            G5_0Psi_h_over_deltastar_avg1p20_jl = AcousticAnalogies.G5_Psi0.(1.20, St_3prime_over_St_3prime_peak_jl)

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_0Psi_h_over_deltastar_avg0p25_jl, St_3prime_over_St_3prime_peak_0p25)
            vmin, vmax = extrema(G5_0Psi_h_over_deltastar_avg0p25)
            err = abs.(interp .- G5_0Psi_h_over_deltastar_avg0p25)/(vmax - vmin)
            @test maximum(err) < 0.030

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_0Psi_h_over_deltastar_avg0p43_jl, St_3prime_over_St_3prime_peak_0p43)
            vmin, vmax = extrema(G5_0Psi_h_over_deltastar_avg0p43)
            err = abs.(interp .- G5_0Psi_h_over_deltastar_avg0p43)/(vmax - vmin)
            @test maximum(err) < 0.026

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_0Psi_h_over_deltastar_avg0p50_jl, St_3prime_over_St_3prime_peak_0p50)
            vmin, vmax = extrema(G5_0Psi_h_over_deltastar_avg0p50)
            err = abs.(interp .- G5_0Psi_h_over_deltastar_avg0p50)/(vmax - vmin)
            @test maximum(err) < 0.037

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_0Psi_h_over_deltastar_avg0p54_jl, St_3prime_over_St_3prime_peak_0p54)
            vmin, vmax = extrema(G5_0Psi_h_over_deltastar_avg0p54)
            err = abs.(interp .- G5_0Psi_h_over_deltastar_avg0p54)/(vmax - vmin)
            @test maximum(err) < 0.037

            # interp = linear(St_3prime_over_St_3prime_peak_jl, G5_0Psi_h_over_deltastar_avg0p62_jl, St_3prime_over_St_3prime_peak_0p62)
            # vmin, vmax = extrema(G5_0Psi_h_over_deltastar_avg0p62)
            # err = abs.(interp .- G5_0Psi_h_over_deltastar_avg0p62)/(vmax - vmin)
            # @test maximum(err) < 0.073

            interp = linear(St_3prime_over_St_3prime_peak_jl, G5_0Psi_h_over_deltastar_avg1p20_jl, St_3prime_over_St_3prime_peak_1p20)
            vmin, vmax = extrema(G5_0Psi_h_over_deltastar_avg1p20)
            err = abs.(interp .- G5_0Psi_h_over_deltastar_avg1p20)/(vmax - vmin)
            @test maximum(err) < 0.045
        end

        @testset "BPM Figure 98b" begin
            # Figures 98 a-d only differ in trailing edge bluntness, so the other sources are all the same.
            # And TBL-TE is the only significant source, other than bluntness.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure98-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # Suction and pressure are the same for zero angle of attack.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure98-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure98-b-bluntness.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_teb_vs = bpm[:, 1]
            SPL_teb_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 60.96e-2  # chord in meters
            U = 69.5  # freestream velocity in m/s
            M = U/340.46
            h = 1.1e-3  # trailing edge bluntness in meters
            Psi = 14*pi/180  # bluntness angle in radians
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)

            SPL_teb_vs_jl = AcousticAnalogies.BLUNT.(f_jl, nu, L, chord, h, Psi, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.053

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            @test maximum(err) < 0.053

            SPL_teb_vs_jl_interp = linear(f_jl, SPL_teb_vs_jl, f_teb_vs.*1e3)
            vmin, vmax = extrema(SPL_teb_vs)
            err = abs.(SPL_teb_vs_jl_interp .- SPL_teb_vs)./(vmax - vmin)
            # Last two points are off.
            # Not sure why.
            @test maximum(err[1:end-2]) < 0.052
            @test maximum(err[1:end-1]) < 0.060
            @test maximum(err) < 0.171
        end

        @testset "BPM Figure 98c" begin
            # Figures 98 a-d only differ in trailing edge bluntness, so the other sources are all the same.
            # And TBL-TE is the only significant source, other than bluntness.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure98-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # Suction and pressure are the same for zero angle of attack.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure98-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure98-c-bluntness.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_teb_vs = bpm[:, 1]
            SPL_teb_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 60.96e-2  # chord in meters
            U = 69.5  # freestream velocity in m/s
            M = U/340.46
            h = 1.9e-3  # trailing edge bluntness in meters
            Psi = 14*pi/180  # bluntness angle in radians
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)

            SPL_teb_vs_jl = AcousticAnalogies.BLUNT.(f_jl, nu, L, chord, h, Psi, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.053

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            @test maximum(err) < 0.053

            SPL_teb_vs_jl_interp = linear(f_jl, SPL_teb_vs_jl, f_teb_vs.*1e3)
            vmin, vmax = extrema(SPL_teb_vs)
            err = abs.(SPL_teb_vs_jl_interp .- SPL_teb_vs)./(vmax - vmin)
            # Last two points are off.
            # Not sure why.
            @test maximum(err[1:end-2]) < 0.040
            @test maximum(err[1:end-1]) < 0.189
            @test err[end] < 0.111
        end

        @testset "BPM Figure 98d" begin
            # Figures 98 a-d only differ in trailing edge bluntness, so the other sources are all the same.
            # And TBL-TE is the only significant source, other than bluntness.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure98-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # Suction and pressure are the same for zero angle of attack.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure98-a-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure98-d-bluntness.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_teb_vs = bpm[:, 1]
            SPL_teb_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 60.96e-2  # chord in meters
            U = 69.5  # freestream velocity in m/s
            M = U/340.46
            h = 2.5e-3  # trailing edge bluntness in meters
            Psi = 14*pi/180  # bluntness angle in radians
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)

            SPL_teb_vs_jl = AcousticAnalogies.BLUNT.(f_jl, nu, L, chord, h, Psi, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.053

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            @test maximum(err) < 0.053

            SPL_teb_vs_jl_interp = linear(f_jl, SPL_teb_vs_jl, f_teb_vs.*1e3)
            vmin, vmax = extrema(SPL_teb_vs)
            err = abs.(SPL_teb_vs_jl_interp .- SPL_teb_vs)./(vmax - vmin)
            # Last two points are off.
            # Not sure why.
            @test maximum(err[1:end-2]) < 0.044
            @test err[end-1] < 0.089
            @test err[end] < 0.089
        end

        @testset "BPM Figure 99b" begin
            # Figures 98 a-d only differ in trailing edge bluntness, so the other sources are all the same.
            # And TBL-TE is the only significant source, other than bluntness.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure99-b-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # Suction and pressure are the same for zero angle of attack.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure99-b-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure99-b-bluntness.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_teb_vs = bpm[:, 1]
            SPL_teb_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 60.96e-2  # chord in meters
            U = 38.6  # freestream velocity in m/s
            M = U/340.46
            h = 1.1e-3  # trailing edge bluntness in meters
            Psi = 14*pi/180  # bluntness angle in radians
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)

            SPL_teb_vs_jl = AcousticAnalogies.BLUNT.(f_jl, nu, L, chord, h, Psi, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.077

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            @test maximum(err) < 0.077

            SPL_teb_vs_jl_interp = linear(f_jl, SPL_teb_vs_jl, f_teb_vs.*1e3)
            vmin, vmax = extrema(SPL_teb_vs)
            err = abs.(SPL_teb_vs_jl_interp .- SPL_teb_vs)./(vmax - vmin)
            # Last two points are off.
            # Not sure why.
            @test maximum(err[1:end-2]) < 0.091
            @test         err[  end-1]  < 0.251
            @test         err[  end  ]  < 0.400
        end

        @testset "BPM Figure 99c" begin
            # Figures 98 a-d only differ in trailing edge bluntness, so the other sources are all the same.
            # And TBL-TE is the only significant source, other than bluntness.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure99-b-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # Suction and pressure are the same for zero angle of attack.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure99-b-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure99-c-bluntness.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_teb_vs = bpm[:, 1]
            SPL_teb_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 60.96e-2  # chord in meters
            U = 38.6  # freestream velocity in m/s
            M = U/340.46
            h = 1.9e-3  # trailing edge bluntness in meters
            Psi = 14*pi/180  # bluntness angle in radians
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)

            SPL_teb_vs_jl = AcousticAnalogies.BLUNT.(f_jl, nu, L, chord, h, Psi, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.077

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            @test maximum(err) < 0.077

            SPL_teb_vs_jl_interp = linear(f_jl, SPL_teb_vs_jl, f_teb_vs.*1e3)
            vmin, vmax = extrema(SPL_teb_vs)
            err = abs.(SPL_teb_vs_jl_interp .- SPL_teb_vs)./(vmax - vmin)
            # Last two points are off.
            # Not sure why.
            @test maximum(err[1:end-2]) < 0.057
            @test         err[  end-1]  < 0.070
            @test         err[  end  ]  < 0.256
        end

        @testset "BPM Figure 99d" begin
            # Figures 98 a-d only differ in trailing edge bluntness, so the other sources are all the same.
            # And TBL-TE is the only significant source, other than bluntness.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure99-b-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_s = bpm[:, 1]
            SPL_s = bpm[:, 2]

            # Suction and pressure are the same for zero angle of attack.
            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure99-b-TBL-TE-suction.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_p = bpm[:, 1]
            SPL_p = bpm[:, 2]

            fname = joinpath(@__DIR__, "bpm_data", "19890016302-figure99-d-bluntness.csv")
            bpm = DelimitedFiles.readdlm(fname, ',')
            f_teb_vs = bpm[:, 1]
            SPL_teb_vs = bpm[:, 2]

            nu = 1.4529e-5  # kinematic viscosity, m^2/s
            L = 45.72e-2  # span in meters
            chord = 60.96e-2  # chord in meters
            U = 38.6  # freestream velocity in m/s
            M = U/340.46
            h = 2.5e-3  # trailing edge bluntness in meters
            Psi = 14*pi/180  # bluntness angle in radians
            r_e = 1.22 # radiation distance in meters
            θ_e = 90*pi/180 
            Φ_e = 90*pi/180
            M_c = 0.8*M
            alphastar = 0.0*pi/180

            f_jl = AcousticMetrics.ExactThirdOctaveCenterBands(0.2e3, 20e3)
            SPL_s_SPL_p_SPL_alpha = AcousticAnalogies.TBL_TE.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha, 2)

            SPL_teb_vs_jl = AcousticAnalogies.BLUNT.(f_jl, nu, L, chord, h, Psi, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.077

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_p)./(vmax - vmin)
            @test maximum(err) < 0.077

            SPL_teb_vs_jl_interp = linear(f_jl, SPL_teb_vs_jl, f_teb_vs.*1e3)
            vmin, vmax = extrema(SPL_teb_vs)
            err = abs.(SPL_teb_vs_jl_interp .- SPL_teb_vs)./(vmax - vmin)
            # Last two points are off.
            # Not sure why.
            @test maximum(err[1:end-3]) < 0.047
            @test err[end-2] < 0.068
            @test err[end-1] < 0.213
            @test err[end] < 0.225
        end

    end

end

end # module