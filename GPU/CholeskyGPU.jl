module CholeskyGPU
include("../utils/RandomMatrix.jl")
using .RandomMatrix
using KernelAbstractions, CUDAKernels, CUDA
using LinearAlgebra


@kernel function dpotrf_kernel!(matrix, output, tilesize, k)
    I, J = @index(Global, NTuple)
    i = tilesize * k - (tilesize - 1)
    j = tilesize * k - (tilesize - 1)
    i_e = tilesize * k
    j_e = tilesize * k

    if I == i && J == j
        value = zero(eltype(output))
        value = sqrt(matrix[I, J])
        output[I, J] = value
    else
        if i <= I <= i_e && j <= J <= j_e && I >= J
            output[I, J] = 1
        end
    end
    # @print("I: ", I, "; endpoint: ", tilesize * k, "\n")
    # @print("J: ", J, "; endpoint: ", tilesize * k, "\n")
end

@kernel function col_kernel!(matrix, output, row_start, col_start)
    I, J = @index(Global, NTuple)
    # first we only change the elements within the currentcol column
    # only affect row elements that are [rowstart, rowend]
    # also only elements that are equal to or greater than J

    # if J == currentcol && rowstart <= I <= rowend && I >= J
    #     if I == J
    #         output[I, J] = sqrt(matrix[I, J])
    #     end
    # end
end

# first step is to create dpotrf
""" 
Given the 'matrix', 'tilesize', and 'k', it computes the cholesky factorization 
of a tilesize x tilesize on the kth diagonal of the matrix.
"""
function dpotrf!(matrix, output, tilesize, k)
    # we need to do column by column
    # the number of columns is tilesize

    i = tilesize * k - (tilesize - 1)
    j = tilesize * k - (tilesize - 1)
    i_e = convert(Int, tilesize * k)
    j_e = convert(Int, tilesize * k)
    for col = j:j_e
        kernel! = col_kernel!(CUDADevice())
        ev = kernel!(matrix, output, i, j)
        wait(ev)
        # kernel! = col_kernel!(CUDADevice())
        # ev = kernel!(matrix, output, j, col, i, i_e, ndrange=(i_e, j_e))
        # wait(ev)
    end
end


function CholeskyFactorization(matrix, output, tilesize=2)
    n, m = size(matrix)

    if n != m
        println("Uh oh, the dimensions of the matrix is not squared")
        return
    end

    t = n / tilesize

    if t % 1 != 0
        println("Tile Size is not allowed")
        return
    end

    for k = 1:t
        println("Current k: ", k)
        # we first calculate the diagonal entries
        dpotrf!(matrix, output, tilesize, k)
    end
end


n = 4
A = RandomHermitianMatrixInt64(n)
juliaImplementation = cholesky(A)

display(A)
display(juliaImplementation.L)

A = CuArray(A)
B = CuArray(zeros(n, n))
CholeskyFactorization(A, B)
display(B)
end
export CholeskyGPU