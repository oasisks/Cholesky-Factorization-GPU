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

        begin
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

@kernel function dtrsm_kernel!(matrix, output, tilesize, k, t, A)
    I_g, J_g = @index(Group, NTuple)
    I, J = @index(Local, NTuple)

    i, j = (I_g * tilesize - (tilesize) + I, J_g * tilesize - (tilesize) + J)

    # we only consider blocks in the kth column
    # and only the column right below the (k, k) tile
    if J_g == k && I_g > k
        tilestart = J_g * tilesize - (tilesize - 1)
        tileend = J_g * tilesize

        temp_sum = zero(eltype(matrix))
        # note all are square matrices
        A_j = convert(Int, j - tilesize * (k - 1))
        for _j = tilestart:tileend
            A_i = convert(Int, _j - tilesize * (k - 1))
            temp_sum += matrix[i, _j] * A[A_i, A_j]
        end

        output[i, j] = temp_sum
    end

end

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
end


"""
This function performs ((A \\ B')') where A is the diagonal tile and the Bs are the column tiles
"""
function dtrsm!(matrix, output, tilesize, k, t)
    # first the 'output' contains the A matrix
    # second, the 'matrix' contains all the B's we need for the columns
    backend = KernelAbstractions.get_backend(matrix)
    kernel! = dtrsm_kernel!(backend)

    # # note this is the makeshift approach
    tile = convert(Int, tilesize * k - (tilesize - 1))
    tile_e = convert(Int, tilesize * k)
    original = view(output, tile:tile_e, tile:tile_e)
    A = inv(original)'
    display(A)
    kernel!(matrix, output, tilesize, k, t, A, ndrange=size(matrix), workgroupsize=(tilesize, tilesize))
    KernelAbstractions.synchronize(backend)
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
        # perform the cholesky on the diagonal tile
        dpotrf!(matrix, output, tilesize, k)

        # now we want to parallize the entire kth block column
        dtrsm!(matrix, output, tilesize, k, t)
        break
    end
end

n = 6
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