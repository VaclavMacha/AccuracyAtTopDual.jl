mutable struct TopPushState{T<:Real}
    s::Vector{T}
    αβ::Vector{T}

    TopPushState(T) = new{T}(T[], T[])
end

Base.@kwdef struct TopPush{S<:Surrogate, T <: Real} <: AbstractTopPush
    C::T = 1
    l::S = Hinge(1)
    state::TopPushState = TopPushState(Float32)
end

find_Δ(a::Real, b::Real, lb::Real, ub::Real) = min(max(lb, - b/a), ub)

# Update rule
function initialization!(model::TopPush, K::KernelMatrix; seed)
    Random.seed!(seed)

    αβ = rand(eltype(K), K.n)
    α, β = projection(αβ[1:K.nα], αβ[(K.nα + 1):end], model.l.ϑ*model.C, 1)
    αβ = vcat(α, β)

    model.state.s = K * αβ
    model.state.αβ = αβ
    return
end

function update!(model::TopPush, K::KernelMatrix, update)
    iszero(update.Δ) && return
    @unpack k, l, Δ, αβ_k, αβ_l = update

    if k <= K.nα && l > K.nα
        model.state.s .+= Δ .* (K[k, :] + K[l, :])
    else
        model.state.s .+= Δ .* (K[k, :] - K[l, :])
    end
    model.state.αβ[k] = αβ_k
    model.state.αβ[l] = αβ_l
    return 
end

# Hinge loss
objective_change(::TopPush{<:Hinge}, ::KernelMatrix, a, b, Δ) = a*Δ^2/2 + b*Δ

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

    a = - K[k, k] + 2*K[k, l] - K[l, l]
    b = - s[k] + s[l]
    lb = max(- αβ[k], αβ[l] - ϑ*C)
    ub = min(ϑ*C - αβ[k], αβ[l])

    Δ = find_Δ(a, b, lb, ub)
    L = objective_change(model, K, a, b, Δ)
    return (; L, Δ, k, l, αβ_k = αβ[k] + Δ, αβ_l = αβ[l] - Δ)
end

function rule_αβ(model::TopPush{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    a = - K[k, k] - 2*K[k, l] - K[l, l]
    b = - s[k] + s[l]
    lb = max(- αβ[k], αβ[l])
    ub = ϑ*C - αβ[k]

    Δ = find_Δ(a, b, lb, ub)
    L = objective_change(model, K, a, b, Δ)
    return (; L, Δ, k, l, αβ_k = αβ[k] + Δ, αβ_l = αβ[l] + Δ)
end

function rule_ββ(model::TopPush{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state

    a = - K[k, k] + 2*K[k, l] - K[l, l]
    b = - s[k] + s[l]
    lb = - αβ[k]
    ub = αβ[l]

    Δ = find_Δ(a, b, lb, ub)
    L = objective_change(model, K, a, b, Δ)
    return (; L, Δ, k, l, αβ_k = αβ[k] + Δ, αβ_l = αβ[l] - Δ)
end

# Quadratic loss
objective_change(::TopPush{<:Quadratic}, ::KernelMatrix, a, b, Δ) = a*Δ^2/2 + b*Δ

function objective(model::TopPush{<:Quadratic}, K::KernelMatrix)
    @unpack s, αβ = model.state
    ϑ, C = model.l.ϑ, model.C

    α = αβ[1:K.nα]
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

    a = - K[k, k] - 2*K[k, l] - K[l, l] - 1/(C*ϑ^2)
    b = - s[k] + s[l] - (αβ[k] - αβ[l])/(2*C*ϑ^2)
    lb = - αβ[k]
    ub = αβ[l]

    Δ = find_Δ(a, b, lb, ub)
    L = objective_change(model, K, a, b, Δ)
    return (; L, Δ, k, l, αβ_k = αβ[k] + Δ, αβ_l = αβ[l] - Δ)
end

function rule_αβ(model::TopPush{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    a = - K[k, k] - 2*K[k, l] - K[l, l] - 1/(2*C*ϑ^2)
    b = - s[k] + s[l]+ 1/ϑ - αβ[k]/(2*C*ϑ^2)
    lb = max(- αβ[k], αβ[l])
    ub = eltype(s)(Inf)

    Δ = find_Δ(a, b, lb, ub)
    L = objective_change(model, K, a, b, Δ)
    return (; L, Δ, k, l, αβ_k = αβ[k] + Δ, αβ_l = αβ[l] + Δ)
end

function rule_ββ(model::TopPush{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state

    a = - K[k, k] + 2*K[k, l] - K[l, l]
    b = - s[k] + s[l]
    lb = - αβ[k]
    ub = αβ[l]

    Δ = find_Δ(a, b, lb, ub)
    L = objective_change(model, K, a, b, Δ)
    return (; L, Δ, k, l, αβ_k = αβ[k] + Δ, αβ_l = αβ[l] - Δ)
end