include("./CholeskyTile.jl")
include("../utils/RandomMatrix.jl")
include("./GPUUtils.jl")

using .CholeskyTile
using .GPUUtils
using Test
using LinearAlgebra
using .RandomMatrix
using CUDA


# @testset "even matrix with an even tilesize = 2" begin
#     n = 8
#     tilesize = 2
#     epsilon = 0.01
#     matrix = RandomHermitianMatrixFloat64(n)
#     display(matrix)
#     display(inv(matrix))
#     juliaImplementation = CuArray(cholesky(matrix).L)
#     matrix = CuArray(matrix)

#     CholeskyTile.CholeskyFactorization(matrix, tilesize)

#     @test GPUUtils.isEqual(matrix, juliaImplementation, epsilon)
# end

# @testset "tilesize = 4" begin
#     n = 12
#     tilesize = 4
#     epsilon = 0.01
#     matrix = RandomHermitianMatrixFloat64(n)
#     println("Generated matrix")
#     display(matrix)
#     display(inv(matrix))
#     juliaImplementation = CuArray(cholesky(matrix).L)
#     matrix = CuArray(matrix)

#     CholeskyTile.CholeskyFactorization(matrix, tilesize)

#     @test GPUUtils.isEqual(matrix, juliaImplementation, epsilon)
# end

# @testset "even matrix with an odd tilesize" begin
#     n = 6
#     tilesize = 3
#     epsilon = 0.01
#     matrix = RandomHermitianMatrixFloat64(n)
#     juliaImplementation = CuArray(cholesky(matrix).L)
#     matrix = CuArray(matrix)

#     CholeskyTile.CholeskyFactorization(matrix, tilesize)

#     @test GPUUtils.isEqual(matrix, juliaImplementation, epsilon)
# end

# @testset "odd matrix with an odd tilesize" begin
#     n = 9
#     tilesize = 3
#     epsilon = 0.01
#     matrix = RandomHermitianMatrixFloat64(n)
#     juliaImplementation = CuArray(cholesky(matrix).L)
#     matrix = CuArray(matrix)

#     CholeskyTile.CholeskyFactorization(matrix, tilesize)

#     @test GPUUtils.isEqual(matrix, juliaImplementation, epsilon)
# end


# @testset "big matrix" begin
#     n = 1000
#     tilesize = 2
#     epsilon = 0.01
#     matrix = RandomHermitianMatrixFloat64(n)

#     juliaImplementation = CuArray(cholesky(matrix).L)
#     matrix = CuArray(matrix)

#     CholeskyTile.CholeskyFactorization(matrix, tilesize)

#     @test GPUUtils.isEqual(matrix, juliaImplementation, epsilon)
# end

@testset "bigger matrix" begin
    n = 22000
    tilesize = 8
    epsilon = 0.01
    println("I am starting to create the matrix")
    matrix = RandomHermitianMatrixFloat64(n)

    # juliaImplementation = CuArray(cholesky(matrix).L)
    println("I am starting to transfer the data from cpu to gpu")
    matrix = CuArray(matrix)

    println("I am doing the calculations")
    CholeskyTile.CholeskyFactorization(matrix, tilesize)

    @test GPUUtils.isEqual(matrix, juliaImplementation, epsilon)
end