module RandomMatrix
using LinearAlgebra

"""
Returns a random hermitian matrix of size 
"""
function RandomHermitianMatrixFloat64(n::Integer)
    if n < 2
        throw(DomainError("The value for n is out of bound for value $n"))
    end
    A = rand(n, n)
    A = (A + A') * 0.5
    A = A + n * I

    return A
end

export RandomHermitianMatrixFloat64
end