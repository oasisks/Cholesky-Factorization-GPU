module NaiveTile
include("../utils/RandomMatrix.jl")
using .RandomMatrix
using LinearAlgebra
using BlockArrays

function dpotrf(matrix::Matrix)
    factorization = cholesky(matrix)
    return Array(factorization.L)
end

function dtrsm(A::Matrix, B::Matrix)
    # return Array((B * (inv(A))'))
    return Array((A \ B')')
end

function dsyrk(A::Matrix, C::Matrix)
    return Array(C - (A * A'))
end

function dgemm(A::Matrix, B::Matrix, C::Matrix)
    # A_ik - A_ij * A_kj'
    return Array(C - A * B')
end

function getTile(i::Int64, j::Int64, M::Matrix, tile_size::Int64)
    # assume square matrices
    tile_i = tile_size * i - (tile_size - 1)
    tile_i_e = tile_size * i

    tile_j = tile_size * j - (tile_size - 1)
    tile_j_e = tile_size * j

    return M[tile_i:tile_i_e, tile_j:tile_j_e]
end

function setTile!(i::Int64, j::Int64, M::Matrix, tile_size::Int64, value::Matrix)
    # assume square matrices
    tile_i = tile_size * i - (tile_size - 1)
    tile_i_e = tile_size * i

    tile_j = tile_size * j - (tile_size - 1)
    tile_j_e = tile_size * j

    M[tile_i:tile_i_e, tile_j:tile_j_e] .= value
end

function TileImplementation(matrix::Matrix)
    # first we want to divide the matrix into 2x2 matrices
    tile_size = 2
    t = trunc(Int, size(matrix, 1) / tile_size)
    for k = 1:t

        # this needs to be done first
        tile = getTile(k, k, matrix, tile_size)
        setTile!(k, k, matrix, tile_size, dpotrf(tile))

        # parallize this next
        for m = k+1:t
            A = getTile(k, k, matrix, tile_size)
            B = getTile(m, k, matrix, tile_size)
            setTile!(m, k, matrix, tile_size, dtrsm(A, B))
        end

        # we wait for the dtrsm to finish its parallel computation
        for n = k+1:t
            A = getTile(n, k, matrix, tile_size)
            C = getTile(n, n, matrix, tile_size)
            setTile!(n, n, matrix, tile_size, dsyrk(A, C))
        end

        # we wait for the dsyrk its parallel computation
        for j = k+1:t
            for i = j+1:t
                A = getTile(i, k, matrix, tile_size)
                B = getTile(j, k, matrix, tile_size)
                C = getTile(i, j, matrix, tile_size)

                setTile!(i, j, matrix, tile_size, dgemm(A, B, C))
            end
        end
    end

    matrix = LowerTriangular(matrix)
end

n = 4
A = RandomHermitianMatrixFloat64(n)
display(A)
c = cholesky(A)

println("Lower triangular")
display(c.L)

# TileImplementation(A)

export TileImplementation
end