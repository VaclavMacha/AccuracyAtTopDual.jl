abstract type Kernel end
struct Linear <: Kernel end
struct Gaussian <: Kernel end

struct KernelType{T<:Real, K}
    γ::T
    scale::Bool
    mlkernels::Bool
    precomputed::Bool

    function KernelType(
        K::Type{<:Kernel};
        γ = 1,
        scale = true,
        T = Float32,
        mlkernels = true,
        precomputed = true,
    )

        return new{T, K}(γ, scale, mlkernels, precomputed)
    end
end

function init(ker::KernelType{T, K}, d) where {T, K}
    γ = compute_gamma(ker, d)
    return ker.mlkernels ? init_mk(K, γ, T) : init_kf(K, γ, T)
end

function compute_gamma(ker::KernelType{T, K}, d) where {T, K}
    return ker.scale ? T(ker.γ/d) : T(ker.γ)
end

# KernelFunctions.jl
function init_kf(::Type{Gaussian}, γ, T)
    K = KernelFunctions.SqExponentialKernel
    return K() ∘ KernelFunctions.ScaleTransform(T(2*γ))
end

init_kf(::Type{Linear}, γ, T) = KernelFunctions.LinearKernel()

function kernelmatrix(kernel::KernelFunctions.Kernel, X)
    return KernelFunctions.kernelmatrix(kernel, X; obsdim = 1)
end

function kernelmatrix(kernel::KernelFunctions.Kernel, X, Y)
    return KernelFunctions.kernelmatrix(kernel, X, Y; obsdim = 1)
end

function kernelmatrix_diag(kernel::KernelFunctions.Kernel, X)
    return KernelFunctions.kernelmatrix_diag(kernel, X; obsdim = 1)
end

# MLKernels.jl
init_mk(::Type{Gaussian}, γ, T) = MLKernels.GaussianKernel{T}(γ)
init_mk(::Type{Linear}, γ, T) = MLKernels.PolynomialKernel{T}(2, 0, 1)

function kernelmatrix(kernel::MLKernels.Kernel, X)
    return MLKernels.kernelmatrix(Val(:row), kernel, X, true)
end

function kernelmatrix(kernel::MLKernels.Kernel, X, Y)
    return MLKernels.kernelmatrix(Val(:row), kernel, X, Y)
end

function kernelmatrix_diag(kernel::MLKernels.Kernel, X)
    return map(x -> MLKernels.kernel(kernel, x, x), eachrow(X))
end

# KernelMatrix
struct KernelMatrix{M, T<:Real, D}
    X::Matrix{T}
    y::BitVector
    kernel::D

    # auxilliary vars
    n::Int
    nα::Int
    nβ::Int
    perm::Vector{Int}

    # kernel matrix
    matrix::M

    function KernelMatrix(model, X::Matrix{T}, y, kernel; precomputed = false) where {T}
        y = BitVector(y)
        nα, nβ, perm = permutation(model, y)

        if precomputed 
            matrix = kernelmatrix(kernel, X[perm, :])
            matrix[1:nα, (nα+1):end] .*= -1
            matrix[(nα+1):end, 1:nα] .*= -1
        else
            diag = kernelmatrix_diag(kernel, X[perm, :])
            matrix = OnFlyMatrix{T}(0, diag, similar(diag))
        end
        D, M = typeof(kernel), typeof(matrix)
        return new{M, T, D}(X, y, kernel, nα + nβ, nα, nβ, perm, matrix)
    end
end

Base.show(io::IO, K::KernelMatrix) = print(io, "$(K.n)x$(K.n) kernel matrix")
Base.size(K::KernelMatrix) = (K.n, K.n)
Base.eltype(::KernelMatrix{M, T}) where {M, T} = T
Base.getindex(K::KernelMatrix, args...) = getindex(K.matrix, args...)
Base.:*(K::KernelMatrix, s::AbstractVector) = K.matrix * s

inds_α(K::KernelMatrix)  = 1:K.nα
inds_β(K::KernelMatrix)  = (K.nα + 1):K.n

# OnFlyMatrix
mutable struct OnFlyMatrix{T<:Real}
    row_id::Int
    diag::Vector{T}
    row::Vector{T}
end

function computerow!(K::KernelMatrix{<:OnFlyMatrix}, i::Int)
    row = kernelmatrix(K.kernel, K.X[K.perm, :], K.X[K.perm[i:i], :])[:] 
    row[i <= K.nα ? inds_β(K) : inds_α(K)] .*= -1

    K.matrix.row .= row 
    K.matrix.row_id = i
    return
end

function Base.getindex(K::KernelMatrix{<:OnFlyMatrix}, i::Int, j::Int)
    if i == j
        K.matrix.diag[i]
    else
        id = K.matrix.row_id
        if i == id
            K.matrix.row[j]
        elseif j == id
            K.matrix.row[i]
        else
            computerow!(K, i)
            K.matrix.row[j]
        end
    end
end

function Base.getindex(K::KernelMatrix{<:OnFlyMatrix}, i::Int, ::Colon)
    i == K.matrix.row_id || computerow!(K, i)
    return copy(K.matrix.row)
end

Base.getindex(K::KernelMatrix{<:OnFlyMatrix}, ::Colon, i::Int) = getindex(K, i, :)

function Base.:*(K::KernelMatrix{<:OnFlyMatrix}, s::AbstractVector)
    x = zero(s)
    for (i, si) in enumerate(s)
        iszero(si) && continue
        x .+= si .* K[i, :]
    end
    return x
end