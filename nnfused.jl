using RxInfer

struct NNFused{E,T} <: DiscreteMultivariateDistribution
    ε::E
    X::T
end;

function train_nn(model, ε, n_iter, train_data, pl_min_dist=0.01)
    optim = Flux.setup(Flux.ADAM(ε), model)
    f = (accuracy) -> -accuracy
    pl = Flux.plateau(f, 5; min_dist=pl_min_dist)
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
        Dense(784, 256, relu),
        Dropout(0.45),
        Dense(256, 256, relu),
        Dropout(0.45),
        Dense(256, 10, relu),
        softmax
    )
    train_data = Flux.DataLoader((fused_neural_net.X, y), shuffle=true, batchsize=128)
    trained_nn = train_nn(model, fused_neural_net.ε, 100, train_data)
    ps = trained_nn(fused_neural_net.X)

    sumlpdf = mean(zip(eachcol(y), eachcol(ps))) do (sy, p)
        return clamp(logpdf(Categorical(p[1:10]), argmax(sy)), -1.0f3, 1.0f3,)
    end

    return sumlpdf
end;

function BayesBase.logpdf(fused_neural_net::NNFused, ε::Real, y::AbstractMatrix{<:Real})
    if ε < 0
        return -1.0f3
    end
    
    model = Chain(
        Dense(784, 256, relu),
        Dropout(0.45),
        Dense(256, 256, relu),
        Dropout(0.45),
        Dense(256, 10, relu),
        softmax
    )
    train_data = Flux.DataLoader((fused_neural_net.X, y), shuffle=true, batchsize=128)
    trained_nn = train_nn(model, ε, 100, train_data)
    ps = trained_nn(fused_neural_net.X)

    sumlpdf = mean(zip(eachcol(y), eachcol(ps))) do (sy, p)
        return clamp(logpdf(Categorical(p[1:10]), argmax(sy)), -1.0f3, 1.0f3,)
    end

    return sumlpdf
end;