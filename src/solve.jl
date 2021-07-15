function update!(model::Model, K::KernelMatrix)
    best = (; L = -Inf, Δ = 0)
    k = rand(1:K.n)
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
        if update.L >= best.L
            best = update
        end
    end
    update!(model, K, best)
    return 
end

function solve!(
    model,
    X,
    y,
    kernel_in;
    type = OnFlyKernel,
    maxiter = 20000,
    seed = 1234,
    scale = true
)   

    kernel = if scale
        with_lengthscale(kernel_in, size(X,2))
    else
        kernel_in
    end

    K = KernelMatrix(model, X, y, kernel; type)

    @info "Initialization"
    initialization!(model, K; seed)

    # progress bar
    Lp0, Ld0, gap0 = objective(model, K)
    bar = ProgressMeter.Progress(maxiter, 0.1, "Training")

    Lps = [Lp0]
    Lds = [Ld0]

    # train
    for _ in 1:maxiter
        update!(model, K)

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
    end

    s = model.state.s
    s[K.inds_αβ[(K.nα+1):end]] .*= -1
    s = s[invperm(K.inds_αβ)]
    return y, s, Lps, Lds
end
