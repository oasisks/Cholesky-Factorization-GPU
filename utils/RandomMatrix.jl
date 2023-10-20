module RandomMatrix
using LinearAlgebra


""" 
Returns a random positive definite hermitian matrix of size n

# Arguments
-   'n::Integer' the size of the matrix. Requires n to be positive. Throws an error if not positive
"""
function RandomHermitianMatrix(n::Integer) 
    if n < 1 
        throw(DomainError("The value for n is out of bound for value $n"))
    end

    A = rand(ComplexF64, n, n)
    A = (A + adjoint(A)) / 2
    eigvals = eigen(A).values

    if !all(eigvals .> 0)
        l_min = minimum(eigvals)
        A += (abs(l_min) + 1.0) * I
    end

    return A
end

export RandomHermitianMatrix
end