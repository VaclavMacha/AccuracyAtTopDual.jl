module AccuracyAtTopDual

using LIBSVM
using ProgressMeter
using Random
using Roots
using UnPack
using Statistics

import MLKernels
import KernelFunctions

using Base.Iterators: partition

abstract type Surrogate end
abstract type Model end
abstract type AbstractTopPush{S<:Surrogate} <: Model end
abstract type AbstractPatMat{S<:Surrogate} <: Model end

function model_name(model::M) where {M<:Model}
    params = values(parameters(model))
    return string(M.name.name, "(", join(params, ", "), ")")
end
Base.show(io::IO, model::Model) = print(io, model_name(model))


export Hinge, Quadratic, KernelType, Linear, Gaussian
export Model, TopPush, TopPushK, τFPL, SVM, PatMat, PatMatNP
export solve!, predict

include("projections.jl")
include("kernels.jl")
include("utilities.jl")
include("toppush.jl")
include("patmat.jl")
include("svm.jl")
include("solve.jl")

end # module
