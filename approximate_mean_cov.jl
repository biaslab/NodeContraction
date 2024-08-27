using RxInfer
using Flux, MLDatasets
using Flux: train!, onehotbatch
using FastGaussQuadrature
using ExponentialFamily
using ProgressMeter
using BayesBase
using ReactiveMP

include("nnfused.jl")

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

# Open a file for writing
open("output.txt", "w") do file
    for i in 3:1000
        m, v = ReactiveMP.approximate_meancov(ghcubature(i), x->pdf(culp,x), right)
        # Write i, m, and v to the file
        write(file, "i: $i, m: $m, v: $v\n")
        @show i, m, v  # This will still print to the console
    end
end

println("Results have been written to output.txt")