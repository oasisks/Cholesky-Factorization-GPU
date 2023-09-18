include("./Naive.jl")
using .Naive

"""
Test a small positive definite hermitian 3x3 matrix
"""
function test1()
    matrix = [4 12 -16; 12 37 -43; -16 -43 98]
    L = [2 0 0; 6 1 0; -8 5 3]
    result = NaiveCholesky(matrix)
    @assert isequal(L, result) "The resulting matrix was not computed correctly"
    print("Passed test1")
end

test1()