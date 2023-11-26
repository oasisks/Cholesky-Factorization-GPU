include("RandomMatrix.jl")
using Test
using LinearAlgebra
using .RandomMatrix


@testset "testing the randomly generated matrices are positive definite hermitian matrices ComplexF64" begin
    @test_throws DomainError RandomHermitianMatrixComplexF64(0)
    @test_throws DomainError RandomHermitianMatrixComplexF64(1)

    tests = [2, 100, 1000]

    for test in tests
        A = RandomHermitianMatrixComplexF64(test)
        @test ishermitian(A)
    end
end

@testset "testing randomly generated matrices that are positive definite hermitian matrices Int64" begin
    @test_throws DomainError RandomHermitianMatrixInt64(0)
    @test_throws DomainError RandomHermitianMatrixInt64(1)

    tests = [2, 100, 1000]

    for test in tests
        A = RandomHermitianMatrixInt64(test)
        @test ishermitian(A)
    end
end