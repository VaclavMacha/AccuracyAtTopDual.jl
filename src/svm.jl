mutable struct SVM{T<:Real} <: Model
    C::T
    state

    SVM(; C::Real = 1, T = Float32) = new{T}(T(C))
end

function Base.show(io::IO, m::SVM)
    print(io, "SVM($(m.C))")
end

function add_params!(solution, model::SVM)
    get!(solution, :model, "SVM")
    get!(solution, :C, model.C)
    return 
end

get_kernel(::Type{Linear}) = LIBSVM.Kernel.Linear
get_kernel(::Type{Gaussian}) = LIBSVM.Kernel.RadialBasis