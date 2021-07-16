Base.broadcastable(c::Surrogate) = Ref(c)

struct Hinge{T<:Real} <: Surrogate
    ϑ::T
end

value(l::Hinge, s) = max(0, 1 + l.ϑ*s)

struct Quadratic{T<:Real} <: Surrogate
    ϑ::T
end

value(l::Quadratic, s) = max(0, 1 + l.ϑ*s)^2