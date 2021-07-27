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

abstract type Model end
abstract type Surrogate end

export Hinge, Quadratic, KernelType, Linear, Gaussian, init
export Model, TopPush, TopPushK, τFPL, SVM
export solve!, predict

include("projections.jl")
include("kernels.jl")
include("surrogates.jl")
include("toppush.jl")
include("svm.jl")
include("solve.jl")

end # module
