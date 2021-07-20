function update!(model::Model, K::KernelMatrix; k  = rand(1:K.n))
    best = (; L = -Inf, Δ = 0)
    K[k,1]
    for l in 1:K.n
        l == k && continue
        update = if k <= K.nα && l <= K.nα
            rule_αα(model, K, k, l)
        elseif k <= K.nα && l > K.nα
            rule_αβ(model, K, k, l)
        elseif k > K.nα && l <= K.nα
            rule_αβ(model, K, l, k)
        else
            rule_ββ(model, K, k, l)
        end
        if update.L > best.L
            best = update
        elseif update.L == best.L
            if rand(Bool)
                best = update
            end
        end
    end
    update!(model, K, best)
    return k == best.l ? (best.l, best.k) : (best.k, best.l)
end

function log!(hist, model, K, iter, k, l; at = 1000)
    T = eltype(K)
    Lprimal, Ldual, gap = objective(model, K)
    append!(get!(hist, :iter, Int[]), iter)
    append!(get!(hist, :k, Int[]), k)
    append!(get!(hist, :l, Int[]), l)
    append!(get!(hist, :Lprimal, T[]), Lprimal)
    append!(get!(hist, :Ldual, T[]), Ldual)
    append!(get!(hist, :gap, T[]), gap)

    if mod(iter, at) == 0
        d = get!(hist, :evolution, Dict{Int, Vector{T}}())
        d[iter] = copy(extract_scores(model, K))
    end
    return
end


function solve!(
    model,
    X::Matrix{T},
    y,
    kernel_in;
    precomputed = false,
    maxiter = 20000,
    seed = 1234,
    scale = true,
    pupdate = 0.9,
    ε::Real = 1e-3,
    at = max(1, round(Int, maxiter/100)),
) where {T<:Real}

    # kernel scaling
    kernel = scale ? kernel_in ∘ ScaleTransform(T(1/size(X,2))) : kernel_in
    K = KernelMatrix(model, X, y, kernel; precomputed)

    # initialization
    @info "Initialization"
    Random.seed!(seed)
    @time initialization!(model, K)

    # progress bar and history
    bar = Progress(maxiter, 1, "Training: ")
    k = rand(1:K.n)
    l = k
    vals = []
    hist = Dict{Symbol, Any}(
        :precomputed => precomputed, 
        :maxiter => maxiter, 
        :seed => seed, 
        :scale => scale, 
        :pupdate => pupdate,
        :ε => ε,
        :labels => y,
    )
    log!(hist, model, K, 0, 0, 0; at)

    # train
    for iter in 1:maxiter
        k = rand() > pupdate ? rand(1:K.n) : l
        k, l = update!(model, K; k)

        # update progress bar
        log!(hist, model, K, iter, k, l; at)
        vals = [
            (:L_primal_0, hist[:Lprimal][1]),
            (:L_dual_0, hist[:Ldual][1]),
            (:gap_0, hist[:gap][1]),
            (:L_primal, hist[:Lprimal][end]),
            (:L_dual, hist[:Ldual][end]),
            (:gap, hist[:gap][end]),
        ]
        next!(bar; showvalues = vals)

        # stop condition
        hist[:gap][end] <= ε && break
    end
    finish!(bar; showvalues = vals)
    hist[:solution] = extract_params(model, K)
    hist[:scores] = extract_scores(model, K)
    return hist
end

function predict(
    model,
    X::Matrix{T},
    y,
    kernel_in,
    Xtest;
    chunksize = 100,
    scale = true,
) where {T<:Real}

    y = BitVector(y)
    nα, ~, perm = permutation(model, y)
    αβ = copy(model.state.αβ)
    αβ[(nα + 1):end] .*= -1

    # kernel scaling
    kernel = scale ? kernel_in ∘ ScaleTransform(T(1/size(X,2))) : kernel_in

    s = zeros(T, size(Xtest, 1))
    for rows in partition(1:size(Xtest, 1), chunksize)
        K = kernelmatrix(kernel, X[perm, :], Xtest[rows, :]; obsdim = 1) 
        s[rows] .= vec(αβ' * K)
    end
    return s
end