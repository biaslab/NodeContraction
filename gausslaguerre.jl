using RxInfer
using Flux, MLDatasets
using Flux: train!, onehotbatch
using FastGaussQuadrature
using ExponentialFamily
using ProgressMeter
using BayesBase

include("nnfused.jl")

BayesBase.prod(::GenericProd, left::ContinuousUnivariateLogPdf, right::GammaDistributionsFamily) = begin
    f = (x) -> exp(logpdf(left,x) + logpdf(right,x) + x)
    x, w = gausslaguerre(400)
    Z = sum(w .* f.(x))
    normalized_f = (x) -> exp(logpdf(left,x) + logpdf(right,x) + x - log(Z))
    expectation_x = sum(w .* normalized_f.(x) .* x)
    expectation_logx = sum(w .* normalized_f.(x) .* log.(x))
    gss = GammaSufficientStatistics(expectation_x, expectation_logx)
    @show gss
    return solve_logpartition_identity(gss, right)
end

struct GammaSufficientStatistics{T}
    x::T
    logx::T
end

function solve_logpartition_identity(statistics::GammaSufficientStatistics, initial_guess::GammaDistributionsFamily)
    f = let statistics = statistics
        (α) -> digamma(α) - log(α / statistics.x) - statistics.logx
    end

    α = find_zero(f, shape(initial_guess), Roots.Order0())

    β = α / statistics.x

    return Gamma(α, inv(β))
end


slice_size = 3000
# Load training data (images, labels)
x_train, y_train = MNIST(split=:train)[:];
x_test, y_test = MNIST(split=:test)[:];
x_test = Flux.flatten(x_test)
# cutted
x_cutted = x_train[:, :, 1:slice_size];
y_cutted = y_train[1:slice_size];

dist = NNFused(1.0f-3, Flux.flatten(x_cutted));
y=Flux.onehotbatch(y_cutted, 0:9);
culp = ContinuousUnivariateLogPdf(UnspecifiedDomain(), (e) -> logpdf(dist, e, y))
right = Gamma(1.0, 1 / 300.0)

@show prod(GenericProd(), culp, right)