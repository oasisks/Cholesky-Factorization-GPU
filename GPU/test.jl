include("./CholeskyTile.jl")
include("../utils/RandomMatrix.jl")
include("./GPUUtils.jl")

using .CholeskyTile
using .GPUUtils
using Test
using LinearAlgebra
using .RandomMatrix
using CUDA


@testset "even matrix with an even tilesize = 2" begin
    n = 8
    tilesize = 2
    epsilon = 0.01
    matrix = RandomHermitianMatrixInt64(n)

    juliaImplementation = CuArray(cholesky(matrix).L)
    matrix = CuArray(matrix)

    CholeskyTile.CholeskyFactorization(matrix, tilesize)

    @test GPUUtils.isEqual(matrix, juliaImplementation, epsilon)
end

@testset "tilesize = 4" begin
    n = 12
    tilesize = 4
    epsilon = 0.01
    matrix = RandomHermitianMatrixInt64(n)
    juliaImplementation = CuArray(cholesky(matrix).L)
    matrix = CuArray(matrix)

    CholeskyTile.CholeskyFactorization(matrix, tilesize)

    display(juliaImplementation)
    display(matrix)
    @test GPUUtils.isEqual(matrix, juliaImplementation, epsilon)
end

@testset "even matrix with an odd tilesize" begin

end

@testset "odd matrix with an odd tilesize" begin

end