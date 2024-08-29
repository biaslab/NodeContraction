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

nuts_10 =  run_ode_experiment(0.01, 0.05, NUTS(0.65), 10)
nuts_100 =  run_ode_experiment(0.01, 0.05, NUTS(0.65), 100)
nuts_1000 = run_ode_experiment(0.01, 0.05, NUTS(0.65), 1000)

sliced_10 = run_ode_experiment(0.01, 0.05, externalsampler(LatentSlice(1)), 10)
sliced_1000 = run_ode_experiment(0.01, 0.05, externalsampler(LatentSlice(1)), 100)
sliced_1000 = run_ode_experiment(0.01, 0.05, externalsampler(LatentSlice(1)), 1000)
