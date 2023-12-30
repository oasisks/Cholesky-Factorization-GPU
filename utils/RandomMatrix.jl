module RandomMatrix
using LinearAlgebra


""" 
Returns a random positive definite hermitian matrix of size n

# Arguments
-   'n::Integer' the size of the matrix. Requires n to be positive. Throws an error if not positive
"""
function RandomHermitianMatrixComplexF64(n::Number)
    if n < 2
        throw(DomainError("The value for n is out of bound for value $n"))
    end

    A = rand(ComplexF64, n, n)
    A = (A + adjoint(A)) / 2
    # display(A)
    eigvals = eigen(A).values
    if !all(eigvals .> 0)
        l_min = minimum(eigvals)
        A += (abs(l_min) + 1) * I
    end

    return A
end

"""
Returns a random hermitian matrix of size 
"""
function RandomHermitianMatrixInt64(n::Integer)
    if n < 2
        throw(DomainError("The value for n is out of bound for value $n"))
    end
    A = rand(Int8, n, n)
    A = (A + adjoint(A)) / 2
    # display(A)
    eigvals = eigen(A).values
    if !all(eigvals .> 0)
        l_min = minimum(eigvals)
        A += (abs(l_min) + 1) * I
    end
    A = trunc.(Int, A)
    return convert(Matrix{Float64}, A)
end

"""
Returns a random hermitian matrix of size 
"""
function RandomHermitianMatrixFloat64(n::Integer)
    if n < 2
        throw(DomainError("The value for n is out of bound for value $n"))
    end
    A = rand(Float64, n, n)
    A = (A + adjoint(A)) / 2
    # display(A)
    eigvals = eigen(A).values
    if !all(eigvals .> 0)
        l_min = minimum(eigvals)
        A += (abs(l_min) + 1) * I
    end

    return A
end

export RandomHermitianMatrixComplexF64, RandomHermitianMatrixInt64, RandomHermitianMatrixFloat64
end