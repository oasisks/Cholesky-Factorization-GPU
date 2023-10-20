include("RandomMatrix.jl")
using Test
using LinearAlgebra
using .RandomMatrix


@testset "testing that the randomly generated matrices are positive definite hermitian matrices" begin
    # RandomHermitianMatrix(0)
    @test RandomHermitianMatrix(0) "Error: it failed to throw an exception for matrix of size 0"

    tests = [1, 100, 1000, 10000, 23123]

    # for test in tests
    #     A = RandomHermitianMatrix(test)
    #     @test ishermitian(A) "Error: it failed for matrices of size $test"
    # end
end