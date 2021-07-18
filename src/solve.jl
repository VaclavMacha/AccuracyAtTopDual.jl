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
    return k == best.l ? best.k : best.l
end

function solve!(
    model,
    X,
    y,
    kernel_in;
    precomputed = false,
    maxiter = 20000,
    seed = 1234,
    scale = true,
    pupdate = 0.9,
    ε::Real = 1e-3,
)   

    kernel = if scale
        with_lengthscale(kernel_in, size(X,2))
    else
        kernel_in
    end

    K = KernelMatrix(model, X, y, kernel; precomputed)

    @info "Initialization"
    Random.seed!(seed)
    initialization!(model, K)

    # progress bar
    Lp0, Ld0, gap0 = objective(model, K)
    bar = ProgressMeter.Progress(maxiter, 1, "Training")

    Lps = [Lp0]
    Lds = [Ld0]
    k = rand(1:K.n)

    # train
    for _ in 1:maxiter
        k = rand() > pupdate ? rand(1:K.n) : k
        k = update!(model, K; k)

        # update progress bar
        Lp, Ld, gap = objective(model, K)
        push!(Lps, Lp)
        push!(Lds, Ld)
        vals = [
            (:L_primal_0, Lp0),
            (:L_dual_0, Ld0),
            (:gap_0, gap0),
            (:L_primal, Lp),
            (:L_dual, Ld),
            (:gap, gap),
        ]
        next!(bar; showvalues = vals)

        gap <= ε && break
    end

    s = extract_scores(model, K)
    return y, s, Lps, Lds
end
