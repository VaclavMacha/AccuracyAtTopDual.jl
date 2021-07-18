module AccuracyAtTopDual

using Reexport

@reexport using KernelFunctions
using UnPack
using ProgressMeter
using Random
using Roots
using Statistics

using Base.Iterators: partition

abstract type Model end
abstract type Surrogate end

export Hinge, Quadratic
export TopPush, TopPushK, τFPL
export solve!, predict

include("projections.jl")
include("kernels.jl")
include("surrogates.jl")
include("toppush.jl")
include("solve.jl")

end # module
