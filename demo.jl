include("contraction.jl")
using RxInfer, GraphPPL

function deterministic_f end

@model function bayes_sir(y)
    i₀ ~ Normal(μ=10.0, σ²=1e2)
    β ~ Normal(μ=10.0, σ²=1e2)

    sol_X ~ deterministic_f(i₀, β)
    for i in eachindex(y)
        pick_sol_X[i] ~ dot(StandardBasisVector(3, i), sol_X)
        # y[i] ~ Poisson(abs(pick_sol_X[i])) # FIXME: This does not work
        y[i] ~ Poisson(l=abs(pick_sol_X[i]))
    end
end;

model = GraphPPL.create_model(bayes_sir()) do model, ctx
    y = GraphPPL.datalabel(model, ctx, GraphPPL.NodeCreationOptions(kind=:data), :y, [1, 2, 3])
    return (y=y,)
end

@show contraction(model)