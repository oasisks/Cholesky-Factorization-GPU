include("./Naive.jl")
using .Naive
using Test
using LinearAlgebra


@testset "small positive definite hermitian 3x3 matrix" begin
    matrix = [4 12 -16; 12 37 -43; -16 -43 98]
    L = convert(Matrix{Float64}, [2 0 0; 6 1 0; -8 5 3])
    result = NaiveCholesky(matrix)
    @test L == result
end