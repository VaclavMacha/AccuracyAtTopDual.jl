struct KernelMatrix{M, T<:Real, D<:Kernel}
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
        nα, nβ, perm = indices(model, y)

        if precomputed 
            matrix = kernelmatrix(kernel, X[perm, :]; obsdim = 1)
            matrix[1:nα, (nα+1):end] .*= -1
            matrix[(nα+1):end, 1:nα] .*= -1
        else
            diag = kernelmatrix_diag(kernel, X[perm, :]; obsdim = 1)
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
    row = kernelmatrix(K.kernel, K.X[K.perm, :], K.X[K.perm[i:i], :]; obsdim = 1)[:] 
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