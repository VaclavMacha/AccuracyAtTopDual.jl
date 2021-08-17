# ------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------
struct TopPushK{S<:Surrogate, T<:Real} <: AbstractTopPush{S}
    K::Int
    C::T
    l::S
    state::State{T}

    function TopPushK(
        K::Int;
        C::Real = 1,
        ϑ::Real = 1,
        surrogate::Type{<:Surrogate} = Hinge,
        T = Float32,
    )
        l = surrogate(T(ϑ))
        return new{typeof(l), T}(K, T(C), l, State(T))
    end
end

function parameters(m::TopPushK{S}) where {S<:Surrogate}
    return (K = m.K, C = m.C, surrogate = S.name.name, ϑ = m.l.ϑ)
end
model_K(model::TopPushK, ::KernelMatrix) = model.K

struct τFPL{S<:Surrogate, T<:Real} <: AbstractTopPush{S}
    τ::T
    C::T
    l::S
    state::State{T}

    function τFPL(
        τ::Real;
        C::Real = 1,
        ϑ::Real = 1,
        surrogate::Type{<:Surrogate} = Hinge,
        T = Float32,
    )
        l = surrogate(T(ϑ))
        return new{typeof(l), T}(T(τ), T(C), l, State(T))
    end
end

function parameters(m::τFPL{S}) where {S<:Surrogate}
    return (τ = m.τ, C = m.C, surrogate = S.name.name, ϑ = m.l.ϑ)
end
model_K(model::τFPL, K::KernelMatrix) = max(1, round(Int, model.τ * K.nβ))

function threshold(model::AbstractTopPush, K::KernelMatrix)
    return - mean(partialsort(model.state.s[inds_β(K)], 1:model_K(model, K)))
end

# TopPush
struct TopPush{S<:Surrogate, T<:Real} <: AbstractTopPush{S}
    C::T
    l::S
    state::State{T}

    function TopPush(;
        C::Real = 1,
        ϑ::Real = 1,
        surrogate::Type{<:Surrogate} = Hinge,
        T = Float32,
    )
        l = surrogate(T(ϑ))
        return new{typeof(l), T}(T(C), l, State(T))
    end
end

function parameters(m::TopPush{S}) where {S<:Surrogate}
    return (C = m.C, surrogate = S.name.name, ϑ = m.l.ϑ)
end
threshold(model::TopPush, K::KernelMatrix) = - minimum(model.state.s[inds_β(K)])

# ------------------------------------------------------------------------------------------
# Auxilliary functions 
# ------------------------------------------------------------------------------------------
function initialization!(model::TopPush, K::KernelMatrix)
    T = eltype(K)

    model.state.s = zeros(T, K.n)
    model.state.αβ = zeros(T, K.n)
    return
end

function initialization!(model::AbstractTopPush{S}, K::KernelMatrix) where {S <: Surrogate}
    T = eltype(K)
    α, β, Km = rand(T, K.nα), zeros(T, K.nβ), model_K(model, K)
    if S <: Hinge
        α, β = projection(α, β, model.l.ϑ*model.C, Km)
    elseif S <: Quadratic
        α, β = projection(α, β, Km)
    else
        throw(ArgumentError("$(S.name.name) surrogate function is not supported"))
    end
    αβ = vcat(α, β)

    model.state.s = K * αβ
    model.state.αβ = αβ
    model.state.αsum = sum(αβ[inds_α(K)])
    model.state.βsort = sort(αβ[inds_β(K)], rev = true)
    return
end

function permutation(::AbstractTopPush, y::BitVector)
    perm_α = findall(y)
    perm_β = findall(.~y)

    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

function extract_scores(model::AbstractTopPush, K::KernelMatrix)
    s = copy(model.state.s)
    s[inds_β(K)] .*= -1
    return s[invperm(K.perm)]
end

function extract_state(model::AbstractTopPush, K::KernelMatrix)
    αβ = copy(model.state.αβ)
    return  (α = αβ[inds_α(K)], β = αβ[inds_β(K)])
end

function objective(model::AbstractTopPush{S}, K::KernelMatrix) where {S<:Surrogate}
    @unpack s, αβ = model.state
    ϑ, C = model.l.ϑ, model.C

    α = αβ[inds_α(K)]
    sα = s[inds_α(K)]
    w_norm = s'*αβ/2
    t = threshold(model, K)

    # objectives
    L_primal = w_norm + C*sum(value.(model.l, t .- sα))
    if S <: Hinge
        L_dual = - w_norm + sum(α)/ϑ
    else
        L_dual = - w_norm + sum(α)/ϑ - sum(abs2, α)/(4*ϑ*C)
    end
    return L_primal, L_dual, L_primal - L_dual
end

# ------------------------------------------------------------------------------------------
# Update rules - Hinge loss
# ------------------------------------------------------------------------------------------
function rule_αα(model::AbstractTopPush{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], αβ[l] - ϑ*C)
    ub = min(ϑ*C - αβ[k], αβ[l])

    return UpdateRule(num, den, lb, ub, k, l)
end

function rule_αβ(model::AbstractTopPush{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, αsum = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] + s[l] - 1/ϑ
    den = K[k, k] + 2*K[k, l] + K[l, l]
    if isa(model, TopPush)
        lb = max(- αβ[k], - αβ[l])
        ub = ϑ*C - αβ[k]
    else
        βmax = find_βmax(model, αβ[l])
        Km = model_K(model, K)
        lb = max(- αβ[k], - αβ[l], Km*βmax - αsum)
        ub = min(ϑ*C - αβ[k], (αsum - Km*αβ[l])/(Km - 1))
    end
    return UpdateRule(num, den, lb, ub, k, l)
end

function rule_ββ(model::AbstractTopPush{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, αsum = model.state

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    if isa(model, TopPush)
        lb = - αβ[k]
        ub = αβ[l]
    else
        Km = model_K(model, K)
        lb = max(- αβ[k], αβ[l] - αsum/Km)
        ub = min(αsum/Km - αβ[k], αβ[l])
    end
    return UpdateRule(num, den, lb, ub, k, l)
end

# ------------------------------------------------------------------------------------------
# Update rules - Quadratic loss
# ------------------------------------------------------------------------------------------
function rule_αα(model::AbstractTopPush{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] - s[l] + (αβ[k] - αβ[l])/(2*C*ϑ^2)
    den = K[k, k] - 2*K[k, l] + K[l, l] + 1/(C*ϑ^2)
    lb = - αβ[k]
    ub = αβ[l]

    return UpdateRule(num, den, lb, ub, k, l)
end

function rule_αβ(model::AbstractTopPush{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, αsum = model.state
    C, ϑ = model.C, model.l.ϑ

    num = s[k] + s[l] - 1/ϑ + αβ[k]/(2*C*ϑ^2)
    den = K[k, k] + 2*K[k, l] + K[l, l] + 1/(2*C*ϑ^2)
    if isa(model, TopPush)
        lb = max(- αβ[k], - αβ[l])
        ub = eltype(s)(Inf)
    else
        βmax = find_βmax(model, αβ[l])
        Km = model_K(model, K)
        lb = max(- αβ[k], - αβ[l], Km*βmax - αsum)
        ub = (αsum - Km*αβ[l])/(Km - 1)
    end
    return UpdateRule(num, den, lb, ub, k, l)
end

function rule_ββ(model::AbstractTopPush{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, αsum = model.state

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    if isa(model, TopPush)
        lb = - αβ[k]
        ub = αβ[l]
    else
        Km = model_K(model, K)
        lb = max(- αβ[k], αβ[l] - αsum/Km)
        ub = min(αsum/Km - αβ[k], αβ[l])
    end
    return UpdateRule(num, den, lb, ub, k, l)
end