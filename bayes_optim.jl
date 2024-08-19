using RxInfer
using Flux, MLDatasets
using Flux: train!, onehotbatch
using ExponentialFamilyProjection
using ProgressMeter
using DataStructures

struct NNFused{E,T} <: DiscreteMultivariateDistribution
    ε::E
    X::T
end;

function train_nn(model, ε, n_iter, train_data)
    optim = Flux.setup(Flux.ADAM(ε), model)
    f = (accuracy) -> -accuracy
    pl = Flux.plateau(f, 5; min_dist=0.01)
    es = Flux.early_stopping(f, 4)

    @showprogress for j in 1:n_iter
        for (i, datapoint) in enumerate(train_data)
            input, labels = datapoint
            val, grads = Flux.withgradient(model) do m
                # Any code inside here is differentiated.
                # Evaluation of the model and loss must be inside!
                result = m(input)
                Flux.logitcrossentropy(result, labels)
            end

            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end
            Flux.update!(optim, model, grads[1])

        end
        accuracy = mean(Flux.onecold(model(train_data.data[1])) .== Flux.onecold(train_data.data[2]))
        if j > 10 && accuracy < 0.75
            break
        end
        pl(accuracy) && break
    end
    return model
end

function BayesBase.logpdf(fused_neural_net::NNFused, y::AbstractMatrix{<:Real})
    model = Chain(
        Dense(784, 100, relu),
        Dropout(0.45),
        Dense(100, 10, relu),
        softmax
    )
    train_data = Flux.DataLoader((fused_neural_net.X, y), shuffle=true, batchsize=128)
    trained_nn = train_nn(model, fused_neural_net.ε, 100, train_data)
    ps = trained_nn(fused_neural_net.X)

    sumlpdf = mean(zip(eachcol(y), eachcol(ps))) do (sy, p)
        return logpdf(Categorical(p[1:10]), argmax(sy))
    end
    if isinf(sumlpdf)
        return -1000.0
    end

    return sumlpdf
end;

slice_size = 3000
# Load training data (images, labels)
x_train, y_train = MNIST(split=:train)[:];
x_test, y_test = MNIST(split=:test)[:];
x_test = Flux.flatten(x_test)
# cutted
x_cutted = x_train[:, :, 1:slice_size];
y_cutted = y_train[1:slice_size];

dist = NNFused(3.0f-2, Flux.flatten(x_cutted))
logpdf(dist, Flux.onehotbatch(y_cutted, 0:9),)

@node NNFused Stochastic [y, ε, X];

@node Exponential Stochastic [out, rate]

@model function bayes_optim(y, X)
    ε ~ Exponential(0.00001)
    y ~ NNFused(ε, X)
end;

@constraints function nn_constraints()
    parameters = ProjectionParameters(
        strategy=ExponentialFamilyProjection.ControlVariateStrategy(nsamples=10),
        niterations=10
    )
    q(ε)::ProjectedTo(Exponential; parameters=parameters)
end;

function ExponentialFamilyProjection.getinitialpoint(::ExponentialFamilyProjection.ControlVariateStrategy, M::ExponentialFamilyProjection.AbstractManifold, parameters::ProjectionParameters)
    return [-100.0]
end

result = infer(
    model=bayes_optim(X=Flux.flatten(x_cutted)),
    data=(y=Flux.onehotbatch(y_cutted, 0:9),),
    constraints=nn_constraints(),
    showprogress=false,
    options=(
        # Read `https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/undefinedrules/`
        rulefallback=NodeFunctionRuleFallback(),
    )
);