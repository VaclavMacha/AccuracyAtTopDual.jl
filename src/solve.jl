function log!(hist, model, K, iter, k, l, Δ; at = 1000)
    T = eltype(K)
    append!(get!(hist, :iter, Int[]), iter)
    append!(get!(hist, :k, Int[]), k)
    append!(get!(hist, :l, Int[]), l)
    append!(get!(hist, :delta, T[]), Δ)
    
    if iter == 0 || Δ != 0
        Lprimal, Ldual, gap = objective(model, K)
        append!(get!(hist, :Lprimal, T[]), Lprimal)
        append!(get!(hist, :Ldual, T[]), Ldual)
        append!(get!(hist, :gap, T[]), gap)
    else
        append!(hist[:Lprimal], hist[:Lprimal][end])
        append!(hist[:Ldual], hist[:Ldual][end])
        append!(hist[:gap], hist[:gap][end])
    end
    
    if mod(iter, at) == 0
        d = get!(hist, :train, Dict{Int, NamedTuple}())
        d[iter] = (; s = extract_scores(model, K), extract_state(model, K)...)
    end
    return
end

function solve!(
    model,
    X::Matrix{T},
    y,
    ker::KernelType{T, D};
    maxiter = 20000,
    seed = 1234,
    pupdate = 0.9,
    ε::Real = 1e-4,
    at = max(1, round(Int, maxiter/100)),
) where {T<:Real, D <: Kernel}

    
    # initialization
    @info "Initialization"
    Random.seed!(seed)
    kernel = init(ker, size(X, 2))
    @time K = KernelMatrix(model, X, y, kernel; precomputed = ker.precomputed)
    @time initialization!(model, K)

    # progress bar and history
    bar = Progress(maxiter, 1, "$(model): ")
    k = rand(1:K.n)
    l = k
    Δ = zero(T)
    vals = []
    hist = Dict{Symbol, Any}(
        :precomputed => ker.precomputed, 
        :maxiter => maxiter, 
        :seed => seed, 
        :kernel => D,
        :γ => ker.γ,
        :scale => ker.scale,
        :mlkernels => ker.mlkernels, 
        :pupdate => pupdate,
        :ε => ε,
    )
    add_params!(hist, model)
    log!(hist, model, K, 0, 0, 0, 0; at)

    # train
    for iter in 1:maxiter
        k = if Δ == 0 || rand() > pupdate
            rand(1:K.n)
        else
            l
        end
        Δ, k, l = update!(model, K; k)

        # update progress bar
        log!(hist, model, K, iter, k, l, Δ; at)
        g0 = hist[:gap][1]
        g = hist[:gap][end]
        vals = [
            (:L_primal, hist[:Lprimal][end]),
            (:L_dual, hist[:Ldual][end]),
            (:gap, g),
            (:iter, iter),
            (:stop, g/g0),
        ]
        next!(bar; showvalues = vals)

        # stop condition
        g/g0 <= ε && break
        isnan(g) && break
        isinf(g) && break
    end
    finish!(bar; showvalues = vals)

    # evaluation
    @info "Evaluation"
    @time hist[:solution] = Dict{Symbol, NamedTuple}(
        :train => (; y, s = extract_scores(model, K), extract_state(model, K)...)
    )
    return hist
end

function predict(
    model,
    X::Matrix{T},
    y,
    ker::KernelType,
    Xtest;
    chunksize = 100,
) where {T<:Real}

    y = BitVector(y)
    αβ = copy(model.state.αβ)
    nα, ~, perm = permutation(model, y)
    αβ[(nα + 1):end] .*= -1
    kernel = init(ker, size(X, 2))

    s = zeros(T, size(Xtest, 1))
    for rows in partition(1:size(Xtest, 1), chunksize)
        K = kernelmatrix(kernel, X[perm, :], Xtest[rows, :])
        s[rows] .= vec(αβ' * K)
    end
    return s
end

# SVM
function solve!(
    model::SVM,
    X::Matrix{T},
    y,
    ker::KernelType{T, D};
    seed = 1234,
    ε::Real = 1e-8,
    kwargs...
) where {T<:Real, D <: Kernel}

    # initialization
    @info "Initialization"
    Random.seed!(seed)
    y_int = Int.(y)
    @time if ker.precomputed
        kernel = init(ker, size(X, 2))
        K = kernelmatrix(kernel, X)
        kernel_type = LIBSVM.Kernel.Precomputed
    else
        K = X'
        kernel_type = get_kernel(D)
    end

    # history
    hist = Dict{Symbol, Any}(
        :precomputed => ker.precomputed, 
        :seed => seed, 
        :kernel => D,
        :γ => ker.γ,
        :scale => ker.scale,
        :mlkernels => ker.mlkernels,
        :ε => ε,
    )
    add_params!(hist, model)

    # train 
    @info "Training"
    @time svm_model = LIBSVM.svmtrain(
        K,
        y_int;
        probability = true,
        cost = Float64(model.C),
        gamma = Float64(compute_gamma(ker, size(X, 2))),
        kernel = kernel_type,
        epsilon = Float64(ε),
    )
    model.state = svm_model

    # evaluation
    @info "Evaluation"
    @time hist[:solution] = Dict{Symbol, NamedTuple}(
        :train => (; y, s = LIBSVM.svmpredict(model.state, K)[2][2, :])
    )
    return hist
end

function predict(
    model::SVM,
    X::Matrix{T},
    y,
    ker::KernelType,
    Xtest;
    chunksize = 100,
) where {T<:Real}

    if ker.precomputed
        kernel = init(ker, size(X, 2))

        s = zeros(T, size(Xtest, 1))
        for rows in partition(1:size(Xtest, 1), chunksize)
            K = kernelmatrix(kernel, X, Xtest[rows, :])
            s[rows] .= LIBSVM.svmpredict(model.state, K)[2][2, :]
        end
    else
        s = LIBSVM.svmpredict(model.state, Xtest')[2][2, :]
    end
    return s
end