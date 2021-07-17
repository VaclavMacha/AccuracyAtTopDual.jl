module AccuracyAtTopDual

using KernelFunctions
using UnPack
using ProgressMeter
using Random
using Roots
using Statistics

abstract type Model end
abstract type Surrogate end

export KernelMatrix, TopPush, solve,PrecomputedKernel, OnFlyKernel, initialization!, update!, solve! ,rule_αα, rule_αβ, rule_ββ, Hinge, Quadratic, TopPushK

include("projections.jl")
include("kernels.jl")
include("surrogates.jl")

# asbtract toppush
abstract type AbstractTopPush <: Model end

function indices(::AbstractTopPush, y::BitVector)
    perm_α = findall(y)
    perm_β = findall(.~y)

    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

struct RuleTopPush{T<:Real}
    L::T
    Δ::T
    num::T
    den::T
    Δlb::T
    Δub::T
    k::Int
    l::Int

    function RuleTopPush(
        ::AbstractTopPush,
        ::KernelMatrix,
        num::T,
        den::Real,
        lb::Real,
        ub::Real,
        k::Int,
        l::Int,
    ) where T
        
        Δ = min(max(lb, - num/den), ub)
        L = - den*Δ^2/2 - num*Δ

        return new{T}(L, Δ, num, den, lb, ub, k, l)
    end
end

function extract_scores(model::AbstractTopPush, K::KernelMatrix)
    s = model.state.s
    s[inds_β(K)] .*= -1
    return s[invperm(K.perm)] 
end

include("toppush.jl")
include("toppushK.jl")
include("solve.jl")

end # module
