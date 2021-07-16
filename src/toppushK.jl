mutable struct TopPushKState{T<:Real}
    s::Vector{T}
    αβ::Vector{T}
    αsum::T
    βsort::Vector{T}

    TopPushKState(T) = new{T}()
end

find_βmax(βsort, βk) = βsort[1] != βk ? βsort[1] : βsort[2]

Base.@kwdef struct TopPushK{S<:Surrogate, T <: Real} <: AbstractTopPush
    K::Int
    C::T = 1
    l::S = Hinge(1)
    state::TopPushKState = TopPushKState(Float32)
end

TopPushK(K; kwargs...) = TopPushK(; K, kwargs...)

function initialization!(model::TopPushK, K::KernelMatrix; seed)
    Random.seed!(seed)

    αβ = rand(eltype(K), K.n)
    α, β = projection(αβ[inds_α(K)], αβ[inds_β(K)], model.l.ϑ*model.C, model.K)
    αβ = vcat(α, β)

    model.state.s = K * αβ
    model.state.αβ = αβ
    model.state.αsum = sum(α)
    model.state.βsort = sort(β, rev = true)
    return
end

function update!(model::TopPushK, K::KernelMatrix, update)
    iszero(update.Δ) && return
    @unpack k, l, Δ = update
    @unpack αβ, βsort = model.state

    y = (k <= K.nα && l > K.nα) ? -1 : 1 
    if k > K.nα
        deleteat!(βsort, searchsortedfirst(βsort, αβ[k]; rev = true))
        insert!(βsort, searchsortedfirst(βsort, αβ[k] + Δ; rev = true), αβ[k] + Δ)
    end
    if l > K.nα
        deleteat!(βsort, searchsortedfirst(βsort, αβ[k]; rev = true))
        insert!(βsort, searchsortedfirst(βsort, αβ[k] - y*Δ; rev = true), αβ[k] - y*Δ)
    end

    model.state.s .+= Δ .* (K[k, :] - y*K[l, :])
    model.state.αβ[k] += Δ
    model.state.αβ[l] -= y*Δ
    if k <= K.nα && l > K.nα
        model.state.αsum += Δ
    end
    return 
end

# Hinge
function objective(model::TopPushK{<:Hinge}, K::KernelMatrix)
    @unpack s, αβ = model.state
    ϑ, C = model.l.ϑ, model.C

    sα = s[inds_α(K)]
    sβ = .- s[inds_β(K)]
    
    w_norm = s'*αβ/2
    t = mean(partialsort(sβ, 1:model.K, rev = true))
    z = max.(sβ .- t, 0)
    y = t + sum(z)/model.K .- sα

    # objectives
    L_primal = w_norm + C*sum(value.(model.l, y))
    L_dual = - w_norm + sum(αβ[inds_α(K)])/ϑ

    return L_primal, L_dual, L_primal - L_dual
end

function rule_αα(model::TopPushK{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], αβ[l] - ϑ*C)
    ub = min(ϑ*C - αβ[k], αβ[l])

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

function rule_αβ(model::TopPushK{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, αsum, βsort = model.state
    C, ϑ = model.C, model.l.ϑ
    βmax = find_βmax(βsort, αβ[l])

    num = s[k] + s[l] - 1/ϑ
    den = K[k, k] + 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], - αβ[l], model.K*βmax - αsum)
    ub = min(ϑ*C - αβ[k], (αsum - model.K*αβ[l])/(model.K - 1))

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

function rule_ββ(model::TopPushK{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, αsum = model.state

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], αβ[l] - αsum/model.K)
    ub = min(αsum/model.K - αβ[k], αβ[l])

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

# Quadratic
function objective(model::TopPushK{<:Quadratic}, K::KernelMatrix)
    @unpack s, αβ = model.state
    ϑ, C = model.l.ϑ, model.C

    α = αβ[inds_α(K)]
    sα = s[inds_α(K)]
    sβ = .- s[inds_β(K)]
    
    w_norm = s'*αβ/2
    t = mean(partialsort(sβ, 1:model.K, rev = true))
    z = max.(sβ .- t, 0)
    y = t + sum(z)/model.K .- sα

    # objectives
    L_primal = w_norm + C*sum(value.(model.l, y))
    L_dual = - w_norm + sum(α)/ϑ - sum(abs2, α)/(4*ϑ*C)

    return L_primal, L_dual, L_primal - L_dual
end

function rule_αα(model::TopPushK{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] - s[l] + (αβ[k] - αβ[l])/(2*C*ϑ^2)
    den = K[k, k] - 2*K[k, l] + K[l, l] + 1/(C*ϑ^2)
    lb = - αβ[k]
    ub = αβ[l]

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

function rule_αβ(model::TopPushK{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, αsum, βsort = model.state
    C, ϑ = model.C, model.l.ϑ
    βmax = find_βmax(βsort, αβ[l])

    num = s[k] + s[l] - 1/ϑ + αβ[k]/(2*C*ϑ^2)
    den = K[k, k] + 2*K[k, l] + K[l, l] + 1/(2*C*ϑ^2)
    lb = max(- αβ[k], - αβ[l], model.K*βmax - αsum)
    ub = (αsum - model.K*αβ[l])/(model.K - 1)

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end

function rule_ββ(model::TopPushK{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, αsum = model.state

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], αβ[l] - αsum/model.K)
    ub = min(αsum/model.K - αβ[k], αβ[l])

    return RuleTopPush(model, K, num, den, lb, ub, k, l)
end