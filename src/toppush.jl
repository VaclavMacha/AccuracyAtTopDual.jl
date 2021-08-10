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

function extract_params(model::AbstractTopPush, K::KernelMatrix)
    αβ = copy(model.state.αβ)
    return  (α = αβ[inds_α(K)], β = αβ[inds_β(K)])
end

mutable struct TPState{T<:Real}
    s::Vector{T}
    αβ::Vector{T}
    αsum::T
    βsort::Vector{T}

    TPState(T) = new{T}()
end

find_βmax(βsort, βk) = βsort[1] != βk ? βsort[1] : βsort[2]

function projection(model::AbstractTopPush{<:Hinge}, K::KernelMatrix, α, β)
    return projection(α, β, model.l.ϑ*model.C, model_K(model, K))
end

function projection(model::AbstractTopPush{<:Quadratic}, K::KernelMatrix, α, β)
    return projection(α, β, model_K(model, K))
end

# ------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------
struct TopPushK{S<:Surrogate, T<:Real} <: AbstractTopPush{S}
    K::Int
    C::T
    l::S
    state::TPState{T}

    function TopPushK(
        K::Int;
        C::Real = 1,
        ϑ::Real = 1,
        surrogate::Type{<:Surrogate} = Hinge,
        T = Float32,
    )
        l = surrogate(T(ϑ))
        return new{typeof(l), T}(K, T(C), l, TPState(T))
    end
end

model_K(model::TopPushK, ::KernelMatrix) = model.K

function Base.show(io::IO, m::TopPushK{S}) where {S<:Surrogate}
    print(io, "TopPushK($(m.K), $(m.C), $(S.name.name), $(m.l.ϑ))")
end

function add_params!(solution, model::TopPushK{S}) where {S<:Surrogate}
    get!(solution, :model, "TopPushK")
    get!(solution, :K, model.K)
    get!(solution, :C, model.C)
    get!(solution, :surrogate, S.name.name)
    get!(solution, :ϑ, model.l.ϑ)
    return 
end

struct τFPL{S<:Surrogate, T<:Real} <: AbstractTopPush{S}
    τ::T
    C::T
    l::S
    state::TPState{T}

    function τFPL(
        τ::Real;
        C::Real = 1,
        ϑ::Real = 1,
        surrogate::Type{<:Surrogate} = Hinge,
        T = Float32,
    )
        l = surrogate(T(ϑ))
        return new{typeof(l), T}(T(τ), T(C), l, TPState(T))
    end
end

model_K(model::τFPL, K::KernelMatrix) = max(1, round(Int, model.τ * K.nβ))

function Base.show(io::IO, m::τFPL{S}) where {S<:Surrogate}
    print(io, "τFPL($(m.τ), $(m.C), $(S.name.name), $(m.l.ϑ))")
end

function add_params!(solution, model::τFPL{S}) where {S<:Surrogate}
    get!(solution, :model, "τFPL")
    get!(solution, :τ, model.τ)
    get!(solution, :C, model.C)
    get!(solution, :surrogate, S.name.name)
    get!(solution, :ϑ, model.l.ϑ)
    return 
end

function threshold(model::AbstractTopPush, K::KernelMatrix)
    return - mean(partialsort(model.state.s[inds_β(K)], 1:model_K(model, K)))
end

function initialization!(model::AbstractTopPush, K::KernelMatrix)
    T = eltype(K)
    α, β = projection(model, K, rand(T, K.nα), zeros(T, K.nβ))
    αβ = vcat(α, β)

    model.state.s = K * αβ
    model.state.αβ = αβ
    model.state.αsum = sum(αβ[inds_α(K)])
    model.state.βsort = sort(αβ[inds_β(K)], rev = true)
    return
end

function update!(model::AbstractTopPush, K::KernelMatrix, update)
    iszero(update.Δ) && return
    @unpack k, l, Δ = update
    @unpack αβ, βsort = model.state

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
    if k <= K.nα && l > K.nα
        model.state.αsum += Δ
    end
    return 
end

# TopPush
struct TopPush{S<:Surrogate, T<:Real} <: AbstractTopPush{S}
    C::T
    l::S
    state::TPState{T}

    function TopPush(;
        C::Real = 1,
        ϑ::Real = 1,
        surrogate::Type{<:Surrogate} = Hinge,
        T = Float32,
    )
        l = surrogate(T(ϑ))
        return new{typeof(l), T}(T(C), l, TPState(T))
    end
end

model_K(::TopPush, ::KernelMatrix) = 1
threshold(model::TopPush, K::KernelMatrix) = - minimum(model.state.s[inds_β(K)])

function Base.show(io::IO, m::TopPush{S}) where {S<:Surrogate}
    print(io, "TopPush($(m.C), $(S.name.name), $(m.l.ϑ))")
end

function add_params!(solution, model::TopPush{S}) where {S<:Surrogate}
    get!(solution, :model, "TopPush")
    get!(solution, :C, model.C)
    get!(solution, :surrogate, S.name.name)
    get!(solution, :ϑ, model.l.ϑ)
    return 
end

function initialization!(model::TopPush, K::KernelMatrix)
    T = eltype(K)

    model.state.s = zeros(T, K.n)
    model.state.αβ = zeros(T, K.n)
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

# ------------------------------------------------------------------------------------------
# Update rules and objectives
# ------------------------------------------------------------------------------------------
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

# Hinge loss
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
        βmax = find_βmax(model.state.βsort, αβ[l])
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

# Quadratic loss
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
        βmax = find_βmax(model.state.βsort, αβ[l])
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