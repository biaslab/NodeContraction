using Distributions
using Turing
using SliceSampling

using ProgressMeter
using OrdinaryDiffEq
using SciMLSensitivity
using Random
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

n_samples = 10000

sampler   = LatentSlice(1)
ode_sliced = sample(bayes_sir(Y), externalsampler(sampler), n_samples);

nsims = 100
i₀_true = 0.01
β_true = 0.05
l = 40
i₀_mean = Array{Float64}(undef, nsims)
β_mean = Array{Float64}(undef, nsims)
i₀_coverage = Array{Float64}(undef, nsims)
β_coverage = Array{Float64}(undef, nsims)

@showprogress for i in 1:nsims
    X_sim, Y_sim = simulate_data(l, i₀_true, β_true)
    r = sample(bayes_sir(Y_sim), externalsampler(sampler), n_samples)
    i₀_mean[i] = mean(r[:i₀])
    i0_cov = sum(r[:i₀] .<= i₀_true) / length(r[:i₀])
    β_mean[i] = mean(r[:β])
    b_cov = sum(r[:β] .<= β_true) / length(r[:β])
    i₀_coverage[i] = i0_cov
    β_coverage[i] = b_cov
end;

# pl_β_coverage = histogram(β_coverage, bins=0:0.1:1.0, label=false, title="β", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
# pl_i₀_coverage = histogram(i₀_coverage, bins=0:0.1:1.0, label=false, title="i₀", ylabel="Density", density=true, xrotation=45, xlim=(0.0,1.0))
# plot(pl_β_coverage, pl_i₀_coverage, layout=(1,2), plot_title="Distribution of CDF of true value")

sum(in_credible_interval.(β_coverage)) / nsims
sum(in_credible_interval.(i₀_coverage)) / nsims