struct History{T<:Real}
    # model setings
    model::Symbol
    params::NamedTuple
    
    # kernel settings
    kernel::Symbol
    γ::T
    scale::Bool
    mlkernels::Bool 
    precomputed::Bool 
    
    # solver settings
    maxiter::Int 
    at::Int
    seed::Int
    pupdate::T
    ε::T

    # history
    iter::Vector{Int}
    k::Vector{Int}
    l::Vector{Int}
    Δ::Vector{T}
    Lprimal::Vector{T}
    Ldual::Vector{T}
    gap::Vector{T}
    train::Dict{Int, NamedTuple}
    solution::Dict{Symbol, NamedTuple}
end

function History(
    model::M,
    ker::KernelType{T, D},
    seed;
    maxiter = 20000,
    at = max(1, round(Int, maxiter/100)),
    pupdate = 0.9,
    ε::Real = 1e-4,
    kwargs...
) where {T<:Real, M<:Model, D<:Kernel}

    bar = Progress(maxiter, 1, "⋅ Training: ")

    return bar, History{T}(
        M.name.name,
        parameters(model),
        D.name.name,
        ker.γ,
        ker.scale,
        ker.mlkernels,
        ker.precomputed, 
        maxiter,
        at,
        seed,
        pupdate,
        ε,
        Int[],
        Int[],
        Int[],
        Int[],
        T[],
        T[],
        T[],
        Dict{Int, NamedTuple}(),
        Dict{Symbol, NamedTuple}(),
    )
end

function log!(h::History, model, K, iter, k, l, Δ)
    append!(h.iter, iter)
    append!(h.k, k)
    append!(h.l, l)
    append!(h.Δ, Δ)
    if Δ != 0 || iter == 0
        Lprimal, Ldual, gap = objective(model, K)
    else
        Lprimal, Ldual, gap = h.Lprimal[end], h.Ldual[end], h.gap[end]
    end
    append!(h.Lprimal, Lprimal)
    append!(h.Ldual, Ldual)
    append!(h.gap, gap)
    if mod(iter, h.at) == 0
        h.train[iter] = (; s = extract_scores(model, K), extract_state(model, K)...)
    end
    return [
        (:L_primal, Lprimal),
        (:L_dual, Ldual),
        (:gap, gap),
        (:iter, iter),
        (:stop, gap/h.gap[1]),
    ]
end

function terminate(h::History)
    g0, g = h.gap[1], h.gap[end]
    return isnan(g) || isinf(g) || g/g0 <= h.ε
end

function solve!(
    model,
    X::Matrix{T},
    y,
    ker::KernelType{T, D};
    seed = 1234,
    kwargs...
) where {T<:Real, D <: Kernel}

    # initialization
    printstyled("$(model): \n"; color=:magenta, bold=true)
    printstyled("⋅ Initialization: "; color=:green)
    tm = @timed begin
        Random.seed!(seed)
        kernel = init(ker, size(X, 2))
        K = KernelMatrix(model, X, y, kernel; precomputed = ker.precomputed)
        initialization!(model, K)
    end
    printstyled(ProgressMeter.durationstring(tm.time), " \n"; color=:green)

    # progress bar and history
    bar, history = History(model, ker, seed; kwargs...)
    k, l, Δ = 0, 0, zero(T)
    vals = log!(history, model, K, 0, k, l, Δ)

    # train
    for iter in 1:history.maxiter
        k = if Δ == 0 || rand() > history.pupdate
            rand(1:K.n)
        else
            l
        end
        Δ, k, l = update!(model, K; k)

        # update progress bar
        vals = log!(history, model, K, iter, k, l, Δ)
        next!(bar; showvalues = vals)

        # stop condition
        terminate(history) && break
    end
    finish!(bar; showvalues = vals)

    # evaluation
    printstyled("⋅ Evaluation: "; color=:green)
    tm = @timed begin
        history.solution[:train] = (;y, s = extract_scores(model, K), extract_state(model, K)...)
    end
    printstyled(ProgressMeter.durationstring(tm.time), " \n\n"; color=:green)
    return history
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
    kwargs...
) where {T<:Real, D <: Kernel}

    # initialization
    printstyled("$(model): \n"; color=:magenta, bold=true)
    printstyled("⋅ Initialization: "; color=:green)
    tm = @timed begin
        Random.seed!(seed)
        if ker.precomputed
            kernel = init(ker, size(X, 2))
            K = kernelmatrix(kernel, X)
            kernel_type = LIBSVM.Kernel.Precomputed
        else
            K = X'
            kernel_type = get_kernel(D)
        end
    end
    printstyled(ProgressMeter.durationstring(tm.time), " \n"; color=:green)

    # train
    printstyled("⋅ Training: "; color=:green)
    tm = @timed begin
        ~, history = History(model, ker, seed; kwargs...)
        svm_model = LIBSVM.svmtrain(
            K,
            Int.(y);
            probability = true,
            cost = Float64(model.C),
            gamma = Float64(compute_gamma(ker, size(X, 2))),
            kernel = kernel_type,
            epsilon = Float64(history.ε),
        )
        model.state = svm_model
    end
    printstyled(ProgressMeter.durationstring(tm.time), " \n"; color=:green)

    # evaluation
    printstyled("⋅ Evaluation: "; color=:green)
    tm = @timed begin
        history.solution[:train] = (;y, s = extract_scores(model, K))
    end
    printstyled(ProgressMeter.durationstring(tm.time), " \n\n"; color=:green)
    return history
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
            s[rows] .= extract_scores(model, K)
        end
    else
        s = extract_scores(model, Xtest)
    end
    return s
end