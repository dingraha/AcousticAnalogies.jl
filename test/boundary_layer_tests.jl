module BoundaryLayerTests

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
            deltastar_jl = AcousticAnalogies.disp_thickness_p.(Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()), alpha_deg_jl.*pi/180)

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
            deltastar_jl = AcousticAnalogies.disp_thickness_s.(Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()), alpha_deg_jl.*pi/180)

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
            deltastar_jl = AcousticAnalogies.disp_thickness_p.(Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()), alpha_deg_jl.*pi/180)

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
            deltastar_jl = AcousticAnalogies.disp_thickness_s.(Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()), alpha_deg_jl.*pi/180)

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
            alphastar0 = 12.5*pi/180
            SPL_s_SPL_p_SPL_alpha_branches = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))
            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 3)
            tblte_branches = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 4)

            @test all(getproperty.(tblte_branches, :K_1) .== 3)
            @test all(getproperty.(tblte_branches, :DeltaK_1) .== 2)
            @test all(getproperty.(tblte_branches, :A_s) .== 3)
            @test all(getproperty.(tblte_branches, :A_p) .== 3)
            @test all(getproperty.(tblte_branches, :B) .== 3)
            @test all(getproperty.(tblte_branches, :St_2) .== 1)
            @test all(getproperty.(tblte_branches, :K_2) .== 1)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.029

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.029

            # Make sure predictions for Figure 11b take all the same branches as Figure 11a.
            U = 55.5  # freestream velocity in m/s
            M = 0.163  # Mach number, corresponds to U = 55.5 m/s in BPM report
            M_c = 0.8*M

            alphastar0 = 12.5*pi/180
            SPL_s_SPL_p_SPL_alpha_branches_11b = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))
            SPL_s_11b_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches_11b, 1)
            SPL_p_11b_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches_11b, 2)
            SPL_alpha_11b_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches_11b, 3)
            tblte_branches_11b = getindex.(SPL_s_SPL_p_SPL_alpha_branches_11b, 4)

            @test all(getproperty.(tblte_branches_11b, :K_1) .== getproperty.(tblte_branches, :K_1))
            @test all(getproperty.(tblte_branches_11b, :DeltaK_1) .== getproperty.(tblte_branches, :DeltaK_1))
            @test all(getproperty.(tblte_branches_11b, :A_s) .== getproperty.(tblte_branches, :A_s))
            @test all(getproperty.(tblte_branches_11b, :A_p) .== getproperty.(tblte_branches, :A_p))
            @test all(getproperty.(tblte_branches_11b, :B) .== getproperty.(tblte_branches, :B))
            @test all(getproperty.(tblte_branches_11b, :St_2) .== getproperty.(tblte_branches, :St_2))
            @test all(getproperty.(tblte_branches_11b, :K_2) .== getproperty.(tblte_branches, :K_2))

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
            alphastar0 = 12.5*pi/180
            SPL_s_SPL_p_SPL_alpha_branches = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))
            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 3)
            tblte_branches = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 4)

            @test all(getproperty.(tblte_branches, :K_1) .== 2)
            @test all(getproperty.(tblte_branches, :DeltaK_1) .== 2)
            @test all(getproperty.(tblte_branches, :A_s) .== 2)
            @test all(getproperty.(tblte_branches, :A_p) .== 2)
            @test all(getproperty.(tblte_branches, :B) .== 2)
            @test all(getproperty.(tblte_branches, :St_2) .== 1)
            @test all(getproperty.(tblte_branches, :K_2) .== 1)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.015

            SPL_p_jl_interp = linear(f_jl, SPL_p_jl, f_p.*1e3)
            vmin, vmax = extrema(SPL_p)
            err = abs.(SPL_p_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.015

            # # Check if all the predictions for Figure 11c also take the same branches as 11d.
            # U = 39.6  # freestream velocity in m/s
            # M = 0.116  # Mach number, corresponds to U = 36.6 m/s in BPM report
            # M_c = 0.8*M

            # alphastar0 = 12.5*pi/180
            # SPL_s_SPL_p_SPL_alpha_branches_11c = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))
            # SPL_s_11c_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches_11c, 1)
            # SPL_p_11c_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches_11c, 2)
            # SPL_alpha_11c_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches_11c, 3)
            # tblte_branches_11c = getindex.(SPL_s_SPL_p_SPL_alpha_branches_11c, 4)

            # @show getproperty.(tblte_branches_11c, :K_1)
            # @show getproperty.(tblte_branches_11c, :DeltaK_1)
            # @show getproperty.(tblte_branches_11c, :A_s)
            # @show getproperty.(tblte_branches_11c, :A_p)
            # @show getproperty.(tblte_branches_11c, :B)
            # @show getproperty.(tblte_branches_11c, :St_2)
            # @show getproperty.(tblte_branches_11c, :K_2)

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
            alphastar0 = 12.5*pi/180
            SPL_s_SPL_p_SPL_alpha_branches = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))
            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 3)
            tblte_branches = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 4)

            @test all(getproperty.(tblte_branches, :K_1) .== 3)
            @test all(getproperty.(tblte_branches, :DeltaK_1) .== 2)
            @test all(getproperty.(tblte_branches, :A_s) .== 3)
            @test all(getproperty.(tblte_branches, :A_p) .== 3)
            @test all(getproperty.(tblte_branches, :B) .== 3)
            @test all(getproperty.(tblte_branches, :St_2) .== 2)
            @test all(getproperty.(tblte_branches, :K_2) .== 2)

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
            alphastar0 = 12.5*pi/180
            SPL_s_SPL_p_SPL_alpha_branches = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 3)
            tblte_branches = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 4)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.015

            @test all(getproperty.(tblte_branches, :K_1) .== 2)
            @test all(getproperty.(tblte_branches, :DeltaK_1) .== 2)
            @test all(getproperty.(tblte_branches, :A_s) .== 2)
            @test all(getproperty.(tblte_branches, :A_p) .== 2)
            @test all(getproperty.(tblte_branches, :B) .== 2)
            @test all(getproperty.(tblte_branches, :St_2) .== 1)
            @test all(getproperty.(tblte_branches, :K_2) .== 1)

            # Inputs for Figure 26b, which doesn't match the report as well as 26a.
            U = 55.5  # freestream velocity in m/s
            M = 0.163  # Mach number, corresponds to U = 55.5 m/s in BPM report
            M_c = 0.8*M
            alphastar0 = 12.5*pi/180
            SPL_s_SPL_p_SPL_alpha_branches_26b = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            # SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches_26b, 1)
            # SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches_26b, 2)
            # SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches_26b, 3)
            tblte_branches_26b = getindex.(SPL_s_SPL_p_SPL_alpha_branches_26b, 4)

            # Check that Figure 26b takes all the same branches as 26a.
            @test all(getproperty.(tblte_branches, :K_1) .== getproperty.(tblte_branches_26b, :K_1))
            @test all(getproperty.(tblte_branches, :DeltaK_1) .== getproperty.(tblte_branches_26b, :DeltaK_1))
            @test all(getproperty.(tblte_branches, :A_s) .== getproperty.(tblte_branches_26b, :A_s))
            @test all(getproperty.(tblte_branches, :A_p) .== getproperty.(tblte_branches_26b, :A_p))
            @test all(getproperty.(tblte_branches, :B) .== getproperty.(tblte_branches_26b, :B))
            @test all(getproperty.(tblte_branches, :St_2) .== getproperty.(tblte_branches_26b, :St_2))
            @test all(getproperty.(tblte_branches, :K_2) .== getproperty.(tblte_branches_26b, :K_2))
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
            alphastar0 = 12.5*pi/180
            SPL_s_SPL_p_SPL_alpha_branches = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.TrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 3)
            tblte_branches = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 4)

            SPL_s_jl_interp = linear(f_jl, SPL_s_jl, f_s.*1e3)
            vmin, vmax = extrema(SPL_s)
            err = abs.(SPL_s_jl_interp .- SPL_s)./(vmax - vmin)
            @test maximum(err) < 0.032

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

            alphastar0 = 12.5*pi/180
            SPL_s_SPL_p_SPL_alpha_branches = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 3)
            tblte_branches = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 4)

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
            alphastar0 = 12.5*pi/180
            SPL_s_SPL_p_SPL_alpha_branches = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))

            SPL_s_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 1)
            SPL_p_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 2)
            SPL_alpha_jl = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 3)
            tblte_branches = getindex.(SPL_s_SPL_p_SPL_alpha_branches, 4)

            @test all(SPL_s_jl .≈ -100)
            @test all(SPL_p_jl .≈ -100)
            @test all(getproperty.(tblte_branches, :K_1) .== 0)
            @test all(getproperty.(tblte_branches, :DeltaK_1) .== 0)
            @test all(getproperty.(tblte_branches, :A_min_a0_p) .== 0)
            @test all(getproperty.(tblte_branches, :A_max_a0_p) .== 0)
            @test all(getproperty.(tblte_branches, :A_min_a_p) .== 0)
            @test all(getproperty.(tblte_branches, :A_max_a_p) .== 0)
            @test all(getproperty.(tblte_branches, :A_p) .== 0)
            @test all(getproperty.(tblte_branches, :B_min_b0) .== 0)
            @test all(getproperty.(tblte_branches, :B_max_b0) .== 0)
            @test all(getproperty.(tblte_branches, :B_min_b) .== 0)
            @test all(getproperty.(tblte_branches, :B_max_b) .== 0)
            @test all(getproperty.(tblte_branches, :B) .== 0)

            SPL_alpha_jl_interp = linear(f_jl, SPL_alpha_jl, f_alpha.*1e3)
            vmin, vmax = extrema(SPL_alpha)
            err = abs.(SPL_alpha_jl_interp .- SPL_alpha)./(vmax - vmin)
            @test maximum(err) < 0.033

            # Now try 69b, which doesn't have as good agreement with the BPM report.
            U = 39.6  # freestream velocity in m/s
            M = 0.116  # Mach number, corresponds to U = 36.6 m/s in BPM report
            SPL_s_SPL_p_SPL_alpha_branches_69b = AcousticAnalogies.TBL_TE_branch.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, alphastar0, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))
            tblte_branches_69b = getindex.(SPL_s_SPL_p_SPL_alpha_branches_69b, 4)
            
            # Test that all the branches taken in 69b are also taken with 69a, which has good aggreement with the BPM report.
            branch_69a = getproperty.(tblte_branches, :A_min_a0_s)
            branch_69b = getproperty.(tblte_branches_69b, :A_min_a0_s)
            @test all(in.(branch_69b, Ref(branch_69a)))

            branch_69a = getproperty.(tblte_branches, :A_max_a0_s)
            branch_69b = getproperty.(tblte_branches_69b, :A_max_a0_s)
            @test all(in.(branch_69b, Ref(branch_69a)))

            branch_69a = getproperty.(tblte_branches, :A_min_a_s)
            branch_69b = getproperty.(tblte_branches_69b, :A_min_a_s)
            @test all(in.(branch_69b, Ref(branch_69a)))

            branch_69a = getproperty.(tblte_branches, :A_max_a_s)
            branch_69b = getproperty.(tblte_branches_69b, :A_max_a_s)
            @test all(in.(branch_69b, Ref(branch_69a)))

            branch_69a = getproperty.(tblte_branches, :A_s)
            branch_69b = getproperty.(tblte_branches_69b, :A_s)
            @test all(in.(branch_69b, Ref(branch_69a)))
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
            alphastar0 = 12.5*pi/180
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
            alphastar0 = 12.5*pi/180
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
            alphastar0 = 12.5*pi/180
            SPL_lbl_vs_jl = AcousticAnalogies.LBL_VS.(f_jl, nu, L, chord, U, M, M_c, r_e, θ_e, Φ_e, alphastar, Ref(AcousticAnalogies.UntrippedN0012BoundaryLayer()))

            SPL_lbl_vs_jl_interp = linear(f_jl, SPL_lbl_vs_jl, f_lbl_vs.*1e3)
            vmin, vmax = extrema(SPL_lbl_vs)
            err = abs.(SPL_lbl_vs_jl_interp .- SPL_lbl_vs)./(vmax - vmin)
            @test maximum(err) < 0.021
        end

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
end

end # module