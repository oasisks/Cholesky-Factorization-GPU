include("./Naive.jl")
include("../utils/RandomMatrix.jl")
include("./NaiveTile.jl")

using .Naive
using .RandomMatrix
using .NaiveTile
using Test
using LinearAlgebra

@testset "small positive definite hermitian 3x3 matrix" begin
    matrix = [4 12 -16; 12 37 -43; -16 -43 98]
    L = convert(Matrix{Float64}, [2 0 0; 6 1 0; -8 5 3])
    result = NaiveCholesky(matrix)
    juliaR = cholesky(matrix)

    @test L == result.L
    @test L == juliaR.L
    @test result.U == juliaR.U
end

@testset "for imaginary numbers" begin
    A = [81 -9im; 9im 45]
    juliaR = cholesky(A)
    result = NaiveCholesky(A)

    @test result.L == juliaR.L
    @test result.U == juliaR.U
end


@testset "generating a random large matrix and test the times" begin 
    n = 1000
    epsilon = 0.01
    A = RandomHermitianMatrixComplexF64(n)

    naive = NaiveCholesky(A)
    juliaImplementation = cholesky(A)

    @test isapprox(naive.L, juliaImplementation.L, atol=epsilon)
    @test isapprox(naive.U, juliaImplementation.U, atol=epsilon)
end

@testset "testing the naive tile implementation of the CPU for even matrices" begin
    n = 8
    epsilon = 0.01
    A = RandomHermitianMatrixInt64(n)
    juliaImplementation = cholesky(A)

    # fix in the future
    A = convert(Matrix{Float64}, A)
    naiveTile = TileImplementation(A)

    @test isapprox(naiveTile, juliaImplementation.L, atol=epsilon)
end