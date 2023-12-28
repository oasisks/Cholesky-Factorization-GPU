module CholeskyTile
include("../utils/RandomMatrix.jl")
# import Pkg    
# Pkg.add(Pkg.PackageSpec(name="KernelAbstractions", version="0.9.14"))
# Pkg.add("CUDA")
using .RandomMatrix
using LinearAlgebra
using KernelAbstractions, CUDA

@kernel function cholesky_kernel!(matrix, output, current_group, current_col, tilesize)
    I_g, J_g = @index(Group, NTuple)
    I, J = @index(Local, NTuple)

    currentI = I_g * tilesize - (tilesize) + I
    currentJ = J_g * tilesize - (tilesize) + J
    # we will do this one column at a time
    if I_g == current_group && J_g == current_group && currentI >= currentJ && currentJ == current_col
        temp_sum = zero(eltype(matrix))

        @inbounds begin
            for _j = currentJ-J+1:currentJ
                temp_sum += output[currentI, _j] * conj(output[currentI, _j])
            end
        end

        # this is on the diagonal
        if I == J
            output[currentI, currentJ] = sqrt(matrix[currentI, currentJ] - temp_sum)
        else
            output[currentI, currentJ] = (1.0 / output[currentJ, currentJ] *
                                          (matrix[currentI, currentJ] - temp_sum))
        end
    end
end

# @kernel function cholesky_kernel!(matrix, output, col_start, current_col, row_start, row_end)
#     I, J = @index(Global, NTuple)

#     if J == current_col && row_start <= I <= row_end && I >= J
#         temp_sum = zero(eltype(matrix))

#         @inbounds begin
#             for _j = 1:J
#                 temp_sum += output[I, _j] * conj(output[I, _j])
#             end
#         end

#         # this is on the diagonal
#         if I == J
#             output[I, J] = sqrt(matrix[I, J] - temp_sum)
#         else
#             output[I, J] = (1.0 / output[J, J] * (matrix[I, J] - temp_sum))
#         end
#     end
# end

"""
This function performs the usual cholesky decomposition algorithm 
"""
function dpotrf!(matrix, output, tilesize, k)
    i = tilesize * k - (tilesize - 1)
    j = tilesize * k - (tilesize - 1)
    i_e = tilesize * k
    j_e = tilesize * k

    backend = KernelAbstractions.get_backend(matrix)
    kernel! = cholesky_kernel!(backend)

    for _j = j:j_e
        kernel!(matrix, output, k, _j, tilesize, ndrange=size(matrix), workgroupsize=(tilesize, tilesize))
        KernelAbstractions.synchronize(backend)
    end
    # for _j = j:j_e
    #     kernel!(matrix, output, j, _j, i, i_e, ndrange=size(matrix))
    #     KernelAbstractions.synchronize(backend)
    # end
end

function CholesktFactorization(matrix, output, tilesize=2)
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
        # println("Current Block ", k)
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

CholesktFactorization(A, B)

display(B)
end
export CholeskyTile