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

export Hinge, Quadratic, KernelType, Linear, Gaussian, init
export Model, TopPush, TopPushK, τFPL, SVM, PatMat, PatMatNP
export solve!, predict

include("projections.jl")
include("kernels.jl")
include("surrogates.jl")
include("update.jl")
include("toppush.jl")
include("patmat.jl")
include("svm.jl")
include("solve.jl")

end # module
