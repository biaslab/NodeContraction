using RxInfer
using Flux, MLDatasets
using Flux: train!, onehotbatch
using ExponentialFamilyProjection
using ProgressMeter

struct NNFused{E, T, M} <: DiscreteMultivariateDistribution
    ε::E
    X::T
    model::M
end;

function train_nn(model, loss, ε, n_iter, train_data)

    optim = Flux.setup(Flux.ADAM(ε), model)
    
    @showprogress for _ in 1:n_iter
        Flux.train!(loss, model, train_data, optim)
    end
    
    return model
end

function BayesBase.logpdf(fused_neural_net::NNFused, y::Vector)

    loss = (m, x, y) -> Flux.Losses.logitcrossentropy(m(x), y);
    
    y_onehot_train = Flux.onehotbatch(y, 0:9)
    train_data = [(Flux.flatten(x_train), Flux.flatten(y_onehot_train))];
    trained_nn = train_nn(fused_neural_net.model, loss, fused_neural_net.ε, 200, train_data)

    ps = trained_nn(train_data[1][1])

    sumlpdf = mean(zip(y, eachcol(ps))) do (y, p)
        return logpdf(Categorical(p[1:10]), y+1)
    end    

    return sumlpdf
end;

slice_size = 10
# Load training data (images, labels)
x_train, y_train = MNIST(split=:train)[:];
# cutted
x_cutted = x_train[:, :, 1:slice_size];
y_cutted = y_train[1:slice_size];

model_generator = () -> Chain(
    Dense(784, 100, relu),
    Dense(100, 10, relu),
    softmax
)

dist = NNFused(1e-5, x_train, model_generator())

# logpdf(dist, y_train)

@node NNFused Stochastic [ y, ε, X, model];

@model function bayes_optim(y, X, model)
    ε ~ Beta(1, 1)
    y ~ NNFused(ε, X, model)
end;

@constraints function nn_constraints()
    parameters = ProjectionParameters(
        strategy = ExponentialFamilyProjection.ControlVariateStrategy(nsamples = 40),
        niterations = 10
    )
    q(ε) :: ProjectedTo(Beta; parameters = parameters)
end;

result = infer(
        model = bayes_sir(X = x_train, model = model_generator() ),
        data  = (y = y_train, ),
        constraints = nn_constraints(),
        showprogress = false,
        options = (
            # Read `https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/undefinedrules/`
            rulefallback = NodeFunctionRuleFallback(),
        )
);