# surrogates
Base.broadcastable(c::Surrogate) = Ref(c)

struct Hinge{T<:Real} <: Surrogate
    ϑ::T
end

value(l::Hinge, s) = max(0, 1 + l.ϑ*s)

struct Quadratic{T<:Real} <: Surrogate
    ϑ::T
end

value(l::Quadratic, s) = max(0, 1 + l.ϑ*s)^2

# model state
mutable struct State{T<:Real}
    s::Vector{T}
    αβ::Vector{T}
    δ::T
    αsum::T
    βsort::Vector{T}

    State(T) = new{T}()
end

find_βmax(βsort, βk) = βsort[1] != βk ? βsort[1] : βsort[2]

# update rules
struct UpdateRule{T<:Real}
    L::T
    Δ::T
    num::T
    den::T
    Δlb::T
    Δub::T
    k::Int
    l::Int
    δ::T
end

UpdateRule(k, l; T = Float32) = UpdateRule{T}(-Inf, 0, 0, 0, 0, 0, k, l, 0)

function UpdateRule(num::T, den::Real, lb::Real, ub::Real, k::Int, l::Int) where T
    Δ = compute_Δ(num, den, lb, ub)
    L = - den*Δ^2/2 - num*Δ

    return UpdateRule{T}(L, Δ, num, den, lb, ub, k, l, 0)
end

function UpdateRule(
    model::AbstractPatMat,
    K::KernelMatrix,
    num::T,
    den::Real,
    lb::Real,
    ub::Real,
    k::Int,
    l::Int,
    δ::Real,
) where T
    
    Δ = compute_Δ(num, den, lb, ub)
    L = - den*Δ^2/2 - num*Δ - δ*K.nβ*model.τ

    return UpdateRule{T}(L, Δ, num, den, lb, ub, k, l, δ)
end

compute_Δ(num, den, lb, ub) = min(max(lb, - num/den), ub)

# Update functions
function select(r1::UpdateRule{T}, r2::UpdateRule{T}; p::Real = 0.5) where T
    r1.Δ == 0 || isinf(r1.L) || isnan(r1.L) && return r2
    r2.Δ == 0 || isinf(r2.L) || isnan(r2.L) && return r1

    return if r1.L == r2.L
        rand() > p ? r1 : r2
    elseif r1.L > r2.L
        r1
    else
        r2
    end
end

function update!(model::Model, K::KernelMatrix; k  = rand(1:K.n))
    best = UpdateRule(k, k; T = eltype(K))
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
        best = select(best, update)
    end
    update!(model, K, best)
    return k == best.l ? (best.Δ, best.l, best.k) : (best.Δ, best.k, best.l)
end
