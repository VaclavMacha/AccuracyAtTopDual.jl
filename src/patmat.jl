# ------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------
struct PatMat{S<:Surrogate, T<:Real} <: AbstractPatMat{S}
    τ::T
    C::T
    l1::S
    l2::S
    state::State{T}

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
        return new{typeof(l1), T}(T(τ), T(C), l1, l2, State(T))
    end
end

struct PatMatNP{S<:Surrogate, T<:Real} <: AbstractPatMat{S}
    τ::T
    C::T
    l1::S
    l2::S
    state::State{T}

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
        return new{typeof(l1), T}(T(τ), T(C), l1, l2, State(T))
    end
end

function parameters(m::AbstractPatMat{S}) where {S<:Surrogate}
    return (m.τ, m.C, S.name.name, m.l1.ϑ, m.l2.ϑ)
end

function threshold(model::AbstractPatMat, K::KernelMatrix)
    s_neg = .- model.state.s[inds_β(K)]
    f(t) = sum(value.(model.l2, s_neg .- t)) - K.nβ*model.τ
  
    return find_root(f, (-Inf, Inf))
end

# ------------------------------------------------------------------------------------------
# Auxilliary functions 
# ------------------------------------------------------------------------------------------
function initialization!(model::AbstractPatMat{S}, K::KernelMatrix) where {S <: Surrogate}
    T = eltype(K)
    α, β = rand(T, K.nα), rand(T, K.nβ)
    if S <: Hinge
        δ = maximum(β)/model.l1.ϑ
        α, β, δ = projection(α, β, δ, model.l1.ϑ*model.C, model.l2.ϑ)
    elseif S <: Quadratic
        δ = sqrt(max(sum(abs2, β)/(4*K.nβ*model.τ*model.l2.ϑ^2), 0))
        α, β, δ = projection(α, β, δ)
    else
        throw(ArgumentError("$(S.name.name) surrogate function is not supported"))
    end
    αβ = vcat(α, β)

    model.state.s = K * αβ
    model.state.αβ = αβ
    model.state.δ = δ
    model.state.βsort = sort(αβ[inds_β(K)], rev = true)
    return
end

function add_params!(solution, model::AbstractPatMat{S}) where {S<:Surrogate}
    get!(solution, :model, typeof(model).name.name)
    get!(solution, :τ, model.τ)
    get!(solution, :C, model.C)
    get!(solution, :surrogate, S.name.name)
    get!(solution, :ϑ1, model.l1.ϑ)
    get!(solution, :ϑ2, model.l2.ϑ)
    return 
end

function permutation(::PatMat, y::BitVector)
    perm_α = findall(y)
    perm_β = 1:length(y)

    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

function permutation(::PatMatNP, y::BitVector)
    perm_α = findall(y)
    perm_β = findall(.~y)

    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

function extract_scores(model::PatMat, K::KernelMatrix)
    s = copy(model.state.s)
    return .- s[inds_β(K)]
end

function extract_scores(model::PatMatNP, K::KernelMatrix)
    s = copy(model.state.s)
    s[inds_β(K)] .*= -1
    return s[invperm(K.perm)] 
end

function extract_state(model::AbstractPatMat, K::KernelMatrix)
    αβ = copy(model.state.αβ)
    return  (α = αβ[inds_α(K)], β = αβ[inds_β(K)], δ = model.state.δ)
end

function objective(model::AbstractPatMat{S}, K::KernelMatrix) where {S<:Surrogate}
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
    if S <: Hinge
        L_dual = - w_norm + sum(α)/ϑ1 + sum(β)/ϑ2 - δ*n*τ
    else
        L_dual = - w_norm + sum(α)/ϑ1 - sum(abs2, α)/(4*C*ϑ1^2) + sum(β)/ϑ2 - sum(abs2, β)/(4*δ*ϑ2^2) - δ*n*τ
    end
    return L_primal, L_dual, L_primal - L_dual
end

# ------------------------------------------------------------------------------------------
# Update rules - Hinge loss
# ------------------------------------------------------------------------------------------
function rule_αα(model::AbstractPatMat{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, δ = model.state
    C, ϑ1 = model.C, model.l1.ϑ

    num = s[k] - s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], αβ[l] - ϑ1*C)
    ub = min(ϑ1*C - αβ[k], αβ[l])

    return UpdateRule(model, K, num, den, lb, ub, k, l, δ)
end

function rule_αβ(model::AbstractPatMat{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    C, ϑ1, ϑ2 = model.C, model.l1.ϑ, model.l2.ϑ
    βmax = find_βmax(model, αβ[l])

    num = s[k] + s[l] - 1/ϑ1 - 1/ϑ2
    den = K[k, k] + 2*K[k, l] + K[l, l]
    lb = max(- αβ[k], - αβ[l])
    ub = ϑ1*C - αβ[k]

    # solution 1
    Δ1 = compute_Δ(num, den, lb, ub)
    r1 = if αβ[l] + Δ1 <= βmax
        UpdateRule(model, K, num, den, lb, ub, k, l, βmax/ϑ2)
    else
        UpdateRule(k, l; T = eltype(K))
    end

    # solution 2
    c = K.nβ*model.τ/ϑ2
    Δ2 = compute_Δ(num + c, den, lb, ub)
    r2 = if αβ[l] + Δ2 >= βmax
        UpdateRule(model, K, num + c, den, lb, ub, k, l, (αβ[l] + Δ2)/ϑ2)
    else
        UpdateRule(k, l; T = eltype(K))
    end
    return select(r1, r2)
end

function rule_ββ(model::AbstractPatMat{<:Hinge}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ = model.state
    ϑ2 = model.l2.ϑ
    βmax = find_βmax(model, αβ[l])

    num = s[k] + s[l]
    den = K[k, k] - 2*K[k, l] + K[l, l]
    lb = - αβ[k]
    ub = αβ[l]

    # solution 1
    Δ1 = compute_Δ(num, den, lb, ub)
    r1 = if max(αβ[k] + Δ1, αβ[l] - Δ1) <= βmax
        UpdateRule(model, K, num, den, lb, ub, k, l, βmax/ϑ2)
    else
        UpdateRule(k, l; T = eltype(K))
    end

    # solution 2
    c = K.nβ*model.τ/ϑ2
    Δ2 = compute_Δ(num + c, den, lb, ub)
    r2 = if αβ[k] + Δ2 >= max(βmax, αβ[l] - Δ2)
        UpdateRule(model, K, num + c, den, lb, ub, k, l, (αβ[k] + Δ2)/ϑ2)
    else
        UpdateRule(k, l; T = eltype(K))
    end

    # solution 3
    Δ3 = compute_Δ(num - c, den, lb, ub)
    r3 = if αβ[l] - Δ3 >= max(βmax, αβ[k] + Δ2)
        UpdateRule(model, K, num - c, den, lb, ub, k, l, (αβ[l] - Δ3)/ϑ2)
    else
        UpdateRule(k, l; T = eltype(K))
    end
    return select(select(r1, r2), r3)
end

# ------------------------------------------------------------------------------------------
# Update rules - Quadratic loss
# ------------------------------------------------------------------------------------------
function rule_αα(model::AbstractPatMat{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, δ = model.state
    C, ϑ1 = model.C, model.l1.ϑ

    num = s[k] - s[l] + (αβ[k] - αβ[l])/(2*C*ϑ1^2)
    den = K[k, k] - 2*K[k, l] + K[l, l] + 1/(C*ϑ1^2)
    lb = - αβ[k]
    ub = αβ[l]

    return UpdateRule(model, K, num, den, lb, ub, k, l, δ)
end

function rule_αβ(model::AbstractPatMat{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, δ = model.state
    C, ϑ1, ϑ2 = model.C, model.l1.ϑ, model.l2.ϑ
    n, τ = K.nβ, model.τ

    num = s[k] + s[l] - 1/ϑ1 + αβ[k]/(2*C*ϑ1^2) - 1/ϑ2 + αβ[l]/(2*δ*ϑ2^2)
    den = K[k, k] + 2*K[k, l] + K[l, l] + 1/(2*C*ϑ2^2) + 1/(2*δ*ϑ2^2)
    lb = max(- αβ[k], - αβ[l])
    ub = eltype(s)(Inf)

    Δ = compute_Δ(num, den, lb, ub)
    δnew = sqrt(max(δ^2 + (Δ^2 + 2*Δ*αβ[l])/(4*ϑ2^2*n*τ), 0))
    num -= (1/δ - 1/δnew)*αβ[l]/(2*ϑ2^2)
    den -= (1/δ - 1/δnew)/(2*ϑ2^2)

    return UpdateRule(model, K, num, den, lb, ub, k, l, δnew)
end

function rule_ββ(model::AbstractPatMat{<:Quadratic}, K::KernelMatrix, k::Int, l::Int)
    @unpack s, αβ, δ = model.state
    ϑ2 = model.l2.ϑ
    n, τ = K.nβ, model.τ

    num = s[k] - s[l] + (αβ[k] - αβ[l])/(2*δ*ϑ2^2)
    den = K[k, k] - 2*K[k, l] + K[l, l] + 1/(δ*ϑ2^2)
    lb = - αβ[k]
    ub = αβ[l]

    Δ = compute_Δ(num, den, lb, ub)
    δnew = sqrt(max(δ^2 + (Δ^2 + Δ*(αβ[k] - αβ[l]))/(2*ϑ2^2*n*τ), 0))
    num -= (1/δ - 1/δnew)*(αβ[k] - αβ[l])/(2*ϑ2^2)
    den -= (1/δ - 1/δnew)/(ϑ2^2)

    return UpdateRule(model, K, num, den, lb, ub, k, l, δnew)
end