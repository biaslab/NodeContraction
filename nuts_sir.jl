using Distributions
using Turing
using SliceSampling

using OrdinaryDiffEq
using SciMLSensitivity
using Random
using ProgressMeter
using DataFrames
using StatsPlots
using BenchmarkTools

include("bayes_sir_turing_spec.jl")

tmax = 40.0
tspan = (0.0,tmax)
obstimes = 1.0:1.0:tmax
u0 = [990.0,10.0,0.0,0.0] # S,I.R,C
p = [0.05,10.0,0.25]; # β,c,γ

prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
sol_ode = solve(prob_ode, Tsit5(), saveat = 1.0);

C = Array(sol_ode)[4,:] # Cumulative cases
X = C[2:end] - C[1:(end-1)];

Random.seed!(1234)
Y = rand.(Poisson.(X));

calls = []
for n_samples in [2, 10, 100, 900, 1000, 10000]
    solver! = SirOdeSolver!(0)
    ode_nuts = sample(bayes_sir(Y, solver!), NUTS(0.65), n_samples, verbose=false, progress=false);
    posterior = DataFrame(ode_nuts);
    append!(calls, solver!.counter)
end 

function run_nuts_cred_intervals(n_samples)

    nsims = 100
    i₀_true = 0.01
    β_true = 0.05
    l = 40
    i₀_mean = Array{Float64}(undef, nsims)
    β_mean = Array{Float64}(undef, nsims)
    i₀_coverage = Array{Float64}(undef, nsims)
    β_coverage = Array{Float64}(undef, nsims)
    number_of_odes_solver = Array{Int}(undef, nsims)

    @showprogress for i in 1:nsims
        X_sim, Y_sim = simulate_data(l, i₀_true, β_true)
        solver! = SirOdeSolver!(0)
        r = sample(bayes_sir(Y_sim, solver!), NUTS(0.65), n_samples, verbose=false, progress=false)
        i₀_mean[i] = mean(r[:i₀])
        i0_cov = sum(r[:i₀] .<= i₀_true) / length(r[:i₀])
        β_mean[i] = mean(r[:β])
        b_cov = sum(r[:β] .<= β_true) / length(r[:β])
        i₀_coverage[i] = i0_cov
        β_coverage[i] = b_cov
        number_of_odes_solver[i] = solver!.counter
    end;

    return mean(i₀_coverage), mean(β_coverage), mean(number_of_odes_solver)
end
Random.seed!(1234)
run_nuts_cred_intervals(10)
Random.seed!(1234)
run_nuts_cred_intervals(100)
Random.seed!(1234)
run_nuts_cred_intervals(900)