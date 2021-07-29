abstract type AbstractPatMat{S<:Surrogate} <: Model end

compute_Δ(num, den, lb, ub) = min(max(lb, - num/den), ub)

struct PMRule{T<:Real}
    L::T
    Δ::T
    num::T
    den::T
    Δlb::T
    Δub::T
    k::Int
    l::Int
    δ::T

    function PMRule(
        model::AbstractPatMat,
        K::KernelMatrix,
        num::T,
        den::Real,
        lb::Real,
        ub::Real,
        k::Int,
        l::Int,
        δ::Real
    ) where T
        
        Δ = compute_Δ(num, den, lb, ub)
        L = - den*Δ^2/2 - num*Δ - δ*K.nβ*model.τ

        return new{T}(L, Δ, num, den, lb, ub, k, l, δ)
    end
end

mutable struct PMState{T<:Real}
    s::Vector{T}
    αβ::Vector{T}
    δ::T
    βsort::Vector{T}

    PMState(T) = new{T}()
end

function projection(model::AbstractPatMat{<:Hinge}, ::KernelMatrix, α, β, δ)
    return projection(α, β, δ, model.l1.ϑ*model.C, model.l2.ϑ)
end

function projection(::AbstractPatMat{<:Quadratic}, ::KernelMatrix, α, β, δ)
    return projection(α, β, δ)
end

function threshold(model::AbstractPatMat, K::KernelMatrix)
    s_neg = .- model.state.s[inds_β(K)]
    f(t) = sum(value.(model.l2, s_neg .- t)) - K.nβ*model.τ
  
    return find_root(f, (-Inf, Inf))
end

# ------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------
struct PatMat{S<:Surrogate, T<:Real} <: AbstractPatMat{S}
    τ::T
    C::T
    l1::S
    l2::S
    state::PMState{T}

    function PatMat(
        τ::Real;
        C::Real = 1,
        ϑ1::Real = 1,
        ϑ2::Real = 1,
        surrogate::Type{<:Surrogate} = Hinge,
        T = Float32,
    )
        l1 = surrogate(T(ϑ1))
        l2 = surrogate(T(ϑ2))
        return new{typeof(l1), T}(T(τ), T(C), l1, l2, PMState(T))
    end
end

function Base.show(io::IO, m::PatMat{S}) where {S<:Surrogate}
    print(io, "PatMat($(m.τ), $(m.C), $(S.name.name), $(m.l1.ϑ), $(m.l2.ϑ))")
end

function permutation(::PatMat, y::BitVector)
    perm_α = findall(y)
    perm_β = 1:length(y)

    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

function extract_scores(model::PatMat, K::KernelMatrix)
    s = copy(model.state.s)
    return .- s[inds_β(K)]
end

function extract_params(model::AbstractPatMat, K::KernelMatrix)
    αβ = copy(model.state.αβ)
    return  (α = αβ[inds_α(K)], β = αβ[inds_β(K)], δ = model.state.δ)
end

function initialization!(model::AbstractPatMat{<:Hinge}, K::KernelMatrix)
    T = eltype(K)
    α, β = rand(T, K.nα), rand(T, K.nβ)
    δ = maximum(β)/model.l1.ϑ
    α, β, δ = projection(model, K, α, β, δ)
    αβ = vcat(α, β)

    model.state.s = K * αβ
    model.state.αβ = αβ
    model.state.δ = δ
    model.state.βsort = sort(αβ[inds_β(K)], rev = true)
    return
end

function initialization!(model::AbstractPatMat{<:Quadratic}, K::KernelMatrix)
    T = eltype(K)
    α, β = rand(T, K.nα), rand(T, K.nβ)
    δ = sqrt(max(sum(abs2, β)/(4*K.nβ*model.τ*model.l2.ϑ^2), 0))
    α, β, δ = projection(model, K, α, β, δ)
    αβ = vcat(α, β)

    model.state.s = K * αβ
    model.state.αβ = αβ
    model.state.δ = δ
    model.state.βsort = sort(αβ[inds_β(K)], rev = true)
    return
end

function update!(model::AbstractPatMat, K::KernelMatrix, update)
    iszero(update.Δ) && return
    @unpack k, l, Δ = update
    @unpack αβ, δ, βsort = model.state

    y = (k <= K.nα && l > K.nα) ? -1 : 1 
    if k > K.nα
        deleteat!(βsort, searchsortedfirst(βsort, αβ[k]; rev = true))
        insert!(βsort, searchsortedfirst(βsort, αβ[k] + Δ; rev = true), αβ[k] + Δ)
    end
    if l > K.nα
        deleteat!(βsort, searchsortedfirst(βsort, αβ[l]; rev = true))
        insert!(βsort, searchsortedfirst(βsort, αβ[l] - y*Δ; rev = true), αβ[l] - y*Δ)
    end

    model.state.s .+= Δ .* (K[k, :] - y*K[l, :])
    model.state.αβ[k] += Δ
    model.state.αβ[l] -= y*Δ
    model.state.δ = δ
    return
end

function add_params!(solution, model::PatMat{S}) where {S<:Surrogate}
    get!(solution, :model, "PatMat")
    get!(solution, :τ, model.τ)
    get!(solution, :C, model.C)
    get!(solution, :surrogate, S.name.name)
    get!(solution, :ϑ1, model.l1.ϑ)
    get!(solution, :ϑ2, model.l2.ϑ)
    return 
end

# PatMatNP
struct PatMatNP{S<:Surrogate, T<:Real} <: AbstractPatMat{S}
    τ::T
    C::T
    l1::S
    l2::S
    state::PMState{T}

    function PatMatNP(
        τ::Real;
        C::Real = 1,
        ϑ1::Real = 1,
        ϑ2::Real = 1,
        surrogate::Type{<:Surrogate} = Hinge,
        T = Float32,
    )
        l1 = surrogate(T(ϑ1))
        l2 = surrogate(T(ϑ2))
        return new{typeof(l1), T}(T(τ), T(C), l1, l2, PMState(T))
    end
end

function Base.show(io::IO, m::PatMatNP{S}) where {S<:Surrogate}
    print(io, "PatMatNP($(m.τ), $(m.C), $(S.name.name), $(m.l1.ϑ), $(m.l2.ϑ))")
end

function permutation(::PatMatNP, y::BitVector)
    perm_α = findall(y)
    perm_β = findall(.~y)

    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

function extract_scores(model::PatMatNP, K::KernelMatrix)
    s = copy(model.state.s)
    s[inds_β(K)] .*= -1
    return s[invperm(K.perm)] 
end

function add_params!(solution, model::PatMatNP{S}) where {S<:Surrogate}
    get!(solution, :model, "PatMatNP")
    get!(solution, :τ, model.τ)
    get!(solution, :C, model.C)
    get!(solution, :surrogate, S.name.name)
    get!(solution, :ϑ1, model.l1.ϑ)
    get!(solution, :ϑ2, model.l2.ϑ)
    return 
end

# ------------------------------------------------------------------------------------------
# Update rules and objectives
# ------------------------------------------------------------------------------------------
# Hinge loss
function objective(model::AbstractPatMat{<:Hinge}, K::KernelMatrix)
    @unpack s, αβ, δ = model.state
    τ = model.τ
    C, ϑ1, ϑ2 = model.C, model.l1.ϑ, model.l2.ϑ

    α = αβ[inds_α(K)]
    β = αβ[inds_β(K)]
    n = K.nβ
    sα = s[inds_α(K)]
    w_norm = s'*αβ/2
    t = threshold(model, K)

    # objectives
    L_primal = w_norm + C*sum(value.(model.l1, t .- sα))
    L_dual = - w_norm + sum(α)/ϑ1 + sum(β)/ϑ2 - δ*n*τ

    return L_primal, L_dual, L_primal - L_dual
end

function rule_αα(model::AbstractPatMat{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, δ = model.state
    C, ϑ1 = model.C, model.l1.ϑ

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], αβ[l] - ϑ1*C)
    ub = min(ϑ1*C - αβ[k], αβ[l])

    return PMRule(model, K, num, den, lb, ub, k, l, δ)
end

function rule_αβ(model::AbstractPatMat{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, βsort = model.state
    C, ϑ1, ϑ2 = model.C, model.l1.ϑ, model.l2.ϑ
    βmax = find_βmax(βsort, αβ[l])

    num = s[k] + s[l] - 1/ϑ1 - 1/ϑ2
    den = K[k, k] + 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], - αβ[l])
    ub = ϑ1*C - αβ[k]

    # solution 1
    Δ1 = compute_Δ(num, den, lb, ub)
    sol1 = if αβ[l] + Δ1 <= βmax
        PMRule(model, K, num, den, lb, ub, k, l, βmax/ϑ2)
    else
        (; L = -Inf, Δ = 0)
    end

    # solution 2
    c = K.nβ*model.τ/ϑ2
    Δ2 = compute_Δ(num + c, den, lb, ub)
    sol2 = if αβ[l] + Δ2 >= βmax
        PMRule(model, K, num + c, den, lb, ub, k, l, (αβ[l] + Δ2)/ϑ2)
    else
        (; L = -Inf, Δ = 0)
    end

    return sol1.L > sol2.L ? sol1 : sol2
end

function rule_ββ(model::AbstractPatMat{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, βsort = model.state
    C, ϑ1, ϑ2 = model.C, model.l1.ϑ, model.l2.ϑ
    βmax = find_βmax(βsort, αβ[l])

    num = s[k] + s[l] - 1/ϑ1 - 1/ϑ2
    den = K[k, k] + 2*K[k, l] + K[l, l]
    lb = - αβ[k]
    ub = αβ[l]

    # solution 1
    Δ1 = compute_Δ(num, den, lb, ub)
    sol1 = if max(αβ[k] + Δ1, αβ[l] - Δ1) <= βmax
        PMRule(model, K, num, den, lb, ub, k, l, βmax/ϑ2)
    else
        (; L = -Inf, Δ = 0)
    end

    # solution 2
    c = K.nβ*model.τ/ϑ2
    Δ2 = compute_Δ(num + c, den, lb, ub)
    sol2 = if αβ[k] + Δ2 >= max(βmax, αβ[l] - Δ2)
        PMRule(model, K, num + c, den, lb, ub, k, l, (αβ[k] + Δ2)/ϑ2)
    else
        (; L = -Inf, Δ = 0)
    end

    # solution 3
    Δ3 = compute_Δ(num - c, den, lb, ub)
    sol3 = if αβ[l] - Δ3 >= max(βmax, αβ[k] + Δ2)
        PMRule(model, K, num - c, den, lb, ub, k, l, (αβ[l] - Δ3)/ϑ2)
    else
        (; L = -Inf, Δ = 0)
    end

    return [sol1, sol2, sol3][argmax([sol1.L, sol2.L, sol3.L])]
end

# Quadratic loss
function objective(model::AbstractPatMat{<:Quadratic}, K::KernelMatrix)
    @unpack s, αβ, δ = model.state
    τ = model.τ
    C, ϑ1, ϑ2 = model.C, model.l1.ϑ, model.l2.ϑ

    α = αβ[inds_α(K)]
    β = αβ[inds_β(K)]
    n = K.nβ
    sα = s[inds_α(K)]
    w_norm = s'*αβ/2
    t = threshold(model, K)

    # objectives
    L_primal = w_norm + C*sum(value.(model.l1, t .- sα))
    L_dual = - w_norm + sum(α)/ϑ1 - sum(abs2, α)/(4*C*ϑ1^2) + sum(β)/ϑ2 - sum(abs2, β)/(4*δ*ϑ2^2) - δ*n*τ

    return L_primal, L_dual, L_primal - L_dual
end

function rule_αα(model::AbstractPatMat{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, δ = model.state
    C, ϑ1 = model.C, model.l1.ϑ

    num = s[k] - s[l] + (αβ[k] - αβ[l])/(2*C*ϑ1^2)
    den = K[k, k] - 2*K[k, l] + K[l, l] + 1/(C*ϑ1^2)
    lb = - αβ[k]
    ub = αβ[l]

    return PMRule(model, K, num, den, lb, ub, k, l, δ)
end

function rule_αβ(model::AbstractPatMat{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, δ = model.state
    C, ϑ1, ϑ2 = model.C, model.l1.ϑ, model.l2.ϑ
    n, τ = K.nβ, model.τ

    num = s[k] + s[l] - 1/ϑ1 + αβ[k]/(2*C*ϑ1^2) - 1/ϑ2 + αβ[l]/(2*δ*ϑ2^2)
    den = K[k, k] + 2*K[k, l] + K[l, l] + 1/(2*C*ϑ2^2)
    lb = max(- αβ[k], - αβ[l])
    ub = eltype(s)(Inf)

    Δ = compute_Δ(num, den, lb, ub)
    δnew = sqrt(max(δ^2 + (Δ^2 + 2*Δ*αβ[l])/(4*ϑ2^2*n*τ), 0))
    num += αβ[l]*(1/δ - 1/δnew)/(2*ϑ2^2)
    den += (1/δ - 1/δnew)/(2*ϑ2^2)

    return PMRule(model, K, num, den, lb, ub, k, l, δnew)
end

function rule_ββ(model::AbstractPatMat{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, δ = model.state
    C, ϑ1, ϑ2 = model.C, model.l1.ϑ, model.l2.ϑ
    n, τ = K.nβ, model.τ

    num = s[k] - s[l] + (αβ[k] - αβ[l])/(2*δ*ϑ2^2)
    den = K[k, k] - 2*K[k, l] + K[l, l] + 1/(2*C*ϑ2^2)
    lb = - αβ[k]
    ub = αβ[l]

    Δ = compute_Δ(num, den, lb, ub)
    δnew = sqrt(max(δ^2 + (Δ^2 + 2*Δ*(αβ[k] - αβ[l]))/(2*ϑ2^2*n*τ), 0))
    num += (1/δ - 1/δnew)*(αβ[k] - αβ[l])/(2*ϑ2^2)
    den += (1/δ - 1/δnew)/(ϑ2^2)

    return PMRule(model, K, num, den, lb, ub, k, l, δnew)
end