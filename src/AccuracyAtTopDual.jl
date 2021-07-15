module AccuracyAtTopDual

using KernelFunctions
using UnPack
using ProgressMeter
using Random
using Roots
using Statistics

abstract type Model end
abstract type AbstractTopPush <: Model end

abstract type Surrogate end

Base.broadcastable(c::Surrogate) = Ref(c)

struct Hinge{T<:Real} <: Surrogate
    ϑ::T
end

value(l::Hinge, s) = max(0, 1 + l.ϑ*s)

struct Quadratic{T<:Real} <: Surrogate
    ϑ::T
end

value(l::Quadratic, s) = max(0, 1 + l.ϑ*s)^2

export KernelMatrix, TopPush, solve,PrecomputedKernel, OnFlyKernel, initialization!, update!, solve! ,rule_αα, rule_αβ, rule_ββ, Hinge, Quadratic

include("projections.jl")
include("utilities.jl")
include("toppush.jl")
include("solve.jl")

end # module
