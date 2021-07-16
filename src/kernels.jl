abstract type KernelMatrix{T<:Real} end

Base.show(io::IO, K::KernelMatrix) = print(io, join(size(K), "x"), " kernel matrix")
Base.eltype(::KernelMatrix{T}) where {T<:Real} = T


inds_α(K::KernelMatrix)  = 1:K.nα
inds_β(K::KernelMatrix)  = (K.nα + 1):K.n
inds_αβ(K::KernelMatrix) = 1:K.nα

# Precomputed
struct PrecomputedKernel{T<:Real, K<:Kernel} <: KernelMatrix{T}
    X::Matrix{T}
    y::BitVector
    kernel::K

    # auxilliary vars
    n::Int
    nα::Int
    nβ::Int
    inds_αβ::Vector{Int}

    # kernel matrix
    K::Matrix{T}

    function PrecomputedKernel(X, y, kernel, inds_α, inds_β)
        nα = length(inds_α)
        inds_αβ = vcat(inds_α, inds_β)
        K = kernelmatrix(kernel, X[inds_αβ, :]; obsdim = 1)
        K[1:nα, (nα+1):end] .*= -1
        K[(nα+1):end, 1:nα] .*= -1

        return new{eltype(K), typeof(kernel)}(
            X,
            BitVector(y),
            kernel,
            length(inds_αβ),
            nα,
            length(inds_β),
            inds_αβ,
            K
        )
    end
end

Base.getindex(K::PrecomputedKernel, args...) = getindex(K.K, args...)
Base.size(K::PrecomputedKernel) = size(K.K)
Base.:*(K::PrecomputedKernel, s::AbstractVector) = K.K*s

# On fly computation
struct OnFlyKernel{T<:Real, K<:Kernel} <: KernelMatrix{T}
    X::Matrix{T}
    y::BitVector
    kernel::K

    # auxilliary vars
    n::Int
    nα::Int
    nβ::Int
    inds_αβ::Vector{Int}

    # kernel matrix
    row_id::Ref{Int}
    row::Vector{T}
    diag::Vector{T}

    function OnFlyKernel(X, y, kernel, inds_α, inds_β)
        nα = length(inds_α)
        inds_αβ = vcat(inds_α, inds_β)
        diag = kernelmatrix_diag(kernel, X[inds_αβ, :]; obsdim = 1)

        return new{eltype(diag), typeof(kernel)}(
            X,
            BitVector(y),
            kernel,
            length(inds_αβ),
            nα,
            length(inds_β),
            inds_αβ,
            Ref(0),
            similar(diag),
            diag
        )
    end
end

function computerow!(K::OnFlyKernel, i::Int)
    row = kernelmatrix(K.kernel, K.X[K.inds_αβ, :], K.X[K.inds_αβ[i:i], :]; obsdim = 1)
    K.row .= row[:] 

    inds = i <= K.nα ? inds_β(K) : inds_α(K)
    K.row[inds] .*= -1
    K.row_id[] = i
    return
end

function Base.getindex(K::OnFlyKernel, i::Int, j::Int)
    if i == j
        K.diag[i]
    else
        if i == K.row_id[]
            K.row[j]
        elseif j == K.row_id[]
            K.row[i]
        else
            computerow!(K, i)
            K.row[j]
        end
    end
end

function Base.getindex(K::OnFlyKernel, i::Int, ::Colon)
    i == K.row_id[] || computerow!(K, i)
    return copy(K.row)
end
Base.size(K::OnFlyKernel) = (K.n, K.n)
function Base.:*(K::OnFlyKernel, s::AbstractVector)
    x = zero(s)
    for (i, si) in enumerate(s)
        iszero(si) && continue
        x .+= si .* K[i, :]
    end
    return x
end
