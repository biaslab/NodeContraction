using RxInfer, GraphPPL

myfilter(::Type{NormalMeanVariance}) = true
myfilter(::Type{Poisson}) = false

function recursive_add(model::GraphPPL.Model, node, current_components, boundary)
    for var_neighbor in GraphPPL.neighbors(model, node)
        for next_factor in GraphPPL.neighbors(model, var_neighbor)
            if next_factor ∉ current_components
                if GraphPPL.NodeBehaviour(model.backend, next_factor.name) == GraphPPL.Deterministic()
                    if var_neighbor ∉ current_components
                        push!(current_components, var_neighbor)
                    end
                    push!(current_components, next_factor)
                    recursive_add(model, next_factor, current_components, boundary)
                else
                    # TODO: @Nimrais implement decent filtering
                    if myfilter(next_factor.name)
                        if var_neighbor ∉ current_components
                            push!(boundary, var_neighbor)
                        end
                    else
                        if var_neighbor ∉ current_components
                            push!(current_components, var_neighbor)
                        end
                        for next_var_neighbor in GraphPPL.neighbors(model, next_factor)
                            if next_var_neighbor ∉ current_components
                                push!(boundary, next_var_neighbor)
                            end
                        end
                    end
                end
            end
        end
    end
end


function contraction(model::GraphPPL.Model)
    clusters = Set()
    for node in GraphPPL.factor_nodes(model)
        if GraphPPL.NodeBehaviour(model.backend, node.name) == GraphPPL.Deterministic() && !any(elem -> node ∈ first(elem), clusters)
            current_cluster = Set()
            boundary = Set()
            recursive_add(model, node, current_cluster, boundary)
            push!(clusters, (current_cluster, boundary))
        end
    end
    return clusters
end