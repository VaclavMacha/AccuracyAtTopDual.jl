mutable struct TopPushState{T<:Real}
    s::Vector{T}
    αβ::Vector{T}

    TopPushState(T) = new{T}(T[], T[])
end

Base.@kwdef struct TopPush{S<:Surrogate, T <: Real, F <: TopPushState} <: AbstractTopPush
    C::T = 1
    l::S = Hinge(1)
    state::F = TopPushState(Float32)
end

function initialization!(model::TopPush, K::KernelMatrix; seed)
    Random.seed!(seed)

    αβ = rand(eltype(K), K.n)
    α, β = projection(αβ[inds_α(K)], αβ[inds_β(K)], model.l.ϑ*model.C, 1)
    αβ = vcat(α, β)

    model.state.s = K * αβ
    model.state.αβ = αβ
    return
end

function update!(model::TopPush, K::KernelMatrix, update)
    iszero(update.Δ) && return
    @unpack k, l, Δ = update

    y = (k <= K.nα && l > K.nα) ? -1 : 1 
    model.state.s .+= Δ .* (K[k, :] - y*K[l, :])
    model.state.αβ[k] += Δ
    model.state.αβ[l] -= y*Δ
    return 
end

# Hinge
function objective(model::TopPush{<:Hinge}, K::KernelMatrix)
    @unpack s, αβ = model.state
    ϑ, C = model.l.ϑ, model.C

    sα = s[inds_α(K)]
    sβ = .- s[inds_β(K)]
    
    w_norm = s'*αβ/2
    t = maximum(sβ)
    z = max.(sβ .- t, 0)
    y = t + sum(z) .- sα

    # objectives
    L_primal = w_norm + C*sum(value.(model.l, y))
    L_dual = - w_norm + sum(αβ[inds_α(K)])/ϑ

    return L_primal, L_dual, L_primal - L_dual
end

function rule_αα(model::TopPush{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], αβ[l] - ϑ*C)
    ub = min(ϑ*C - αβ[k], αβ[l])

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

function rule_αβ(model::TopPush{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] + s[l] - 1/ϑ
    den = K[k, k] + 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], - αβ[l])
    ub = ϑ*C - αβ[k]

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

function rule_ββ(model::TopPush{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = - αβ[k]
    ub = αβ[l]

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

# Quadratic
function objective(model::TopPush{<:Quadratic}, K::KernelMatrix)
    @unpack s, αβ = model.state
    ϑ, C = model.l.ϑ, model.C

    α = αβ[inds_α(K)]
    sα = s[inds_α(K)]
    sβ = .- s[inds_β(K)]
    
    w_norm = s'*αβ/2
    t = maximum(sβ)
    z = max.(sβ .- t, 0)
    y = t + sum(z) .- sα

    # objectives
    L_primal = w_norm + C*sum(value.(model.l, y))
    L_dual = - w_norm + sum(α)/ϑ - sum(abs2, α)/(4*ϑ*C)

    return L_primal, L_dual, L_primal - L_dual
end

function rule_αα(model::TopPush{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] - s[l] + (αβ[k] - αβ[l])/(2*C*ϑ^2)
    den = K[k, k] - 2*K[k, l] + K[l, l] + 1/(C*ϑ^2)
    lb = - αβ[k]
    ub = αβ[l]

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

function rule_αβ(model::TopPush{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] + s[l] - 1/ϑ + αβ[k]/(2*C*ϑ^2)
    den = K[k, k] + 2*K[k, l] + K[l, l] + 1/(2*C*ϑ^2)
    lb = max(- αβ[k], - αβ[l])
    ub = eltype(s)(Inf)

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

function rule_ββ(model::TopPush{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = - αβ[k]
    ub = αβ[l]

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end