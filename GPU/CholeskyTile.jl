module CholeskyTile
include("../utils/RandomMatrix.jl")
# import Pkg    
# Pkg.add(Pkg.PackageSpec(name="KernelAbstractions", version="0.9.14"))
# Pkg.add("CUDA")
using .RandomMatrix
using LinearAlgebra
using KernelAbstractions, CUDA

@kernel function cholesky_kernel!(matrix, output, k, current_col, tilesize)
    I_g, J_g = @index(Group, NTuple)
    I, J = @index(Local, NTuple)

    i = tilesize * (I_g - 1) + I
    j = tilesize * (J_g - 1) + J
    # we will do this one column at a time
    if I_g == k && J_g == k && i >= j && j == current_col
        temp_sum = zero(eltype(matrix))
        s = convert(Int, (k - 1) * tilesize + 1)
        for _j = s:j-1
            temp_sum += output[i, _j] * conj(output[i, _j])
        end

        # this is to deal with the base case
        elt_ij = k == 1 ? matrix[i, j] : output[i, j]
        # this is on the diagonal
        if I == J
            # @print("matrix value: ", matrix[i, j], "; sum = ", temp_sum, "sqrt() = ", sqrt(matrix[i, j] - temp_sum), "\n")
            output[i, j] = sqrt(elt_ij - temp_sum)
        else
            output[i, j] = (1.0 / output[j, j] *
                            (elt_ij - temp_sum))
        end
    end

    if I_g == k && J_g == k && i < j && j == current_col
        output[i, j] = 0
    end
end

@kernel function dtrsm_kernel!(matrix, output, tilesize, k, A)
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

@kernel function dsyrk_kernel!(matrix, output, tilesize, k)
    I_g, J_g = @index(Group, NTuple)
    I, J = @index(Local, NTuple)

    i, j = (I_g * tilesize - (tilesize) + I, J_g * tilesize - (tilesize) + J)

    # we want to consider all Diagonal groups after the (k, k)
    if I_g == J_g && I_g > k
        # we want to first calculate A * A' where A is the column value in the 'output'
        # the values of this matrix exists within block 'k'
        s_j = convert(Int, (k - 1) * tilesize + 1)
        e_j = convert(Int, k * tilesize)

        AA_t = zero(eltype(output))

        # this is A * A'
        for _j = s_j:e_j
            AA_t += output[i, _j] * output[j, _j]
        end

        output[i, j] = matrix[i, j] - AA_t
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
function dtrsm!(matrix, output, tilesize, k)
    # first the 'output' contains the A matrix
    # second, the 'matrix' contains all the B's we need for the columns
    backend = KernelAbstractions.get_backend(matrix)
    kernel! = dtrsm_kernel!(backend)

    # # note this is the makeshift approach
    tile = convert(Int, tilesize * k - (tilesize - 1))
    tile_e = convert(Int, tilesize * k)
    original = view(output, tile:tile_e, tile:tile_e)
    A = inv(original)'
    kernel!(matrix, output, tilesize, k, A, ndrange=size(matrix), workgroupsize=(tilesize, tilesize))
    KernelAbstractions.synchronize(backend)
end

"""
This function performs C - (A * A') where C is the diagonal matrix and A is the column matrix
"""
function dsyrk!(matrix, output, tilesize, k)
    backend = KernelAbstractions.get_backend(matrix)
    kernel! = dsyrk_kernel!(backend)

    kernel!(matrix, output, tilesize, k, ndrange=size(matrix), workgroupsize=(tilesize, tilesize))
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
        println("After performing DPOTRF in iteration: ", k)
        display(output)
        println()

        # now we want to parallize the entire kth block column
        dtrsm!(matrix, output, tilesize, k)
        println("After performing DTRSM in iteration: ", k)
        display(output)
        println()

        # now we want to parallize all the diagonals
        dsyrk!(matrix, output, tilesize, k)
        println("After performing DYSRK in iteration: ", k)
        display(output)
        println()
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

end
export CholeskyTile