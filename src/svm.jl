mutable struct SVM{T<:Real} <: Model
    C::T
    state

    SVM(; C::Real = 1, T = Float32) = new{T}(T(C))
end

get_kernel(::Type{Linear}) = LIBSVM.Kernel.Linear
get_kernel(::Type{Gaussian}) = LIBSVM.Kernel.RadialBasis
parameters(m::SVM) = (C = m.C, )
extract_scores(model::SVM, K) = LIBSVM.svmpredict(model.state, K)[2][2, :]
