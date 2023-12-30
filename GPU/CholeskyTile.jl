module CholeskyTile
include("../utils/RandomMatrix.jl")
# import Pkg    
# Pkg.add(Pkg.PackageSpec(name="KernelAbstractions", version="0.9.14"))
# Pkg.add("CUDA")
using .RandomMatrix
using LinearAlgebra
using KernelAbstractions, CUDA

@kernel function cholesky_kernel!(matrix, k, current_col, tilesize)
    I_g, J_g = @index(Group, NTuple)
    I, J = @index(Local, NTuple)

    i = tilesize * (I_g - 1) + I
    j = tilesize * (J_g - 1) + J
    # we will do this one column at a time
    if I_g == k && J_g == k && i >= j && j == current_col
        temp_sum = zero(eltype(matrix))
        s = convert(Int, (k - 1) * tilesize + 1)
        for _j = s:j-1
            temp_sum += matrix[i, _j] * conj(matrix[i, _j])
        end

        # diagonal
        if I == J
            # @print("matrix value: ", matrix[i, j], "; sum = ", temp_sum, "sqrt() = ", sqrt(matrix[i, j] - temp_sum), "\n")
            matrix[i, j] = sqrt(matrix[i, j] - temp_sum)
        else
            matrix[i, j] = (1.0 / matrix[j, j] *
                            (matrix[i, j] - temp_sum))
        end
    end

    if i < j
        matrix[i, j] = 0
    end
end

@kernel function dtrsm_kernel!(matrix, tilesize, k, A)
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

        matrix[i, j] = temp_sum
    end
end

@kernel function dsyrk_kernel!(matrix, tilesize, k)
    I_g, J_g = @index(Group, NTuple)
    I, J = @index(Local, NTuple)

    i, j = (I_g * tilesize - (tilesize) + I, J_g * tilesize - (tilesize) + J)

    # we want to consider all Diagonal groups after the (k, k)
    if I_g == J_g && I_g > k
        # we want to first calculate A * A' where A is the column value in the 'output'
        # the values of this matrix exists within block 'k'
        s_j = convert(Int, (k - 1) * tilesize + 1)
        e_j = convert(Int, k * tilesize)

        AA_t = zero(eltype(matrix))

        # this is A * A'
        for _j = s_j:e_j
            AA_t += matrix[i, _j] * matrix[j, _j]
        end

        # this is C - A * A'
        matrix[i, j] = matrix[i, j] - AA_t
    end
end
"""
This function performs C - A * B' in parallel where C is located in row i column j
    A = Matrix[i, k] and B = Matrix[j, k]
"""
@kernel function dgemm_kernel!(matrix, tilesize, k)
    I_g, J_g = @index(Group, NTuple)
    I, J = @index(Local, NTuple)

    i, j = (I_g * tilesize - (tilesize) + I, J_g * tilesize - (tilesize) + J)

    # We want to consider all of the groups that are not on any diagonals
    # Also groups that are not on the Kth column
    if J_g > k && I_g > J_g
        # to update an element (i, j), we first grab the element C
        c = matrix[i, j]

        # calculate A * B'
        ab_t = zero(eltype(matrix))
        A_s = convert(Int, (k - 1) * tilesize + 1)
        A_e = convert(Int, k * tilesize)

        for _j = A_s:A_e
            # this is the row elt of A
            a = matrix[i, _j]
            b = matrix[j, _j]
            ab_t += a * b
        end

        matrix[i, j] = c - ab_t
    end
end

"""
This function performs the usual cholesky decomposition algorithm 
"""
function dpotrf!(matrix, tilesize, k)
    i = tilesize * k - (tilesize - 1)
    j = tilesize * k - (tilesize - 1)
    i_e = tilesize * k
    j_e = tilesize * k

    backend = KernelAbstractions.get_backend(matrix)
    kernel! = cholesky_kernel!(backend)

    for _j = j:j_e
        kernel!(matrix, k, _j, tilesize, ndrange=size(matrix), workgroupsize=(tilesize, tilesize))
        KernelAbstractions.synchronize(backend)
    end
end


"""
This function performs ((A \\ B')') where A is the diagonal tile and the Bs are the column tiles
"""
function dtrsm!(matrix, tilesize, k)
    # first the 'output' contains the A matrix
    # second, the 'matrix' contains all the B's we need for the columns
    backend = KernelAbstractions.get_backend(matrix)
    kernel! = dtrsm_kernel!(backend)

    # # note this is the makeshift approach
    tile = convert(Int, tilesize * k - (tilesize - 1))
    tile_e = convert(Int, tilesize * k)
    original = view(matrix, tile:tile_e, tile:tile_e)
    A = inv(original)'
    kernel!(matrix, tilesize, k, A, ndrange=size(matrix), workgroupsize=(tilesize, tilesize))
    KernelAbstractions.synchronize(backend)
end

"""
This function performs C - (A * A') where C is the diagonal matrix and A is the column matrix
"""
function dsyrk!(matrix, tilesize, k)
    backend = KernelAbstractions.get_backend(matrix)
    kernel! = dsyrk_kernel!(backend)

    kernel!(matrix, tilesize, k, ndrange=size(matrix), workgroupsize=(tilesize, tilesize))
    KernelAbstractions.synchronize(backend)
end

"""
This function performs C - A * B' where C is located in row m column n 
    A = Matrix[m, k] and B = Matrix[n, k]
"""
function dgemm!(matrix, tilesize, k)
    backend = KernelAbstractions.get_backend(matrix)
    kernel! = dgemm_kernel!(backend)

    kernel!(matrix, tilesize, k, ndrange=size(matrix), workgroupsize=(tilesize, tilesize))
    KernelAbstractions.synchronize(backend)
end


function CholesktFactorization(matrix, tilesize=2)
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
        dpotrf!(matrix, tilesize, k)
        # println("After performing DPOTRF in iteration: ", k)
        # display(matrix)
        # println()

        # now we want to parallize the entire kth block column
        dtrsm!(matrix, tilesize, k)
        # println("After performing DTRSM in iteration: ", k)
        # display(matrix)
        # println()

        # now we want to parallize all the diagonals
        dsyrk!(matrix, tilesize, k)
        # println("After performing DYSRK in iteration: ", k)
        # display(matrix)
        println()

        # now we want to deal with the rest of the tiles
        dgemm!(matrix, tilesize, k)
        # println("After performing DGEMM in iteration: ", k)
        # display(matrix)
        # println()

    end

end

n = 8
A = RandomHermitianMatrixInt64(n)
juliaImplementation = cholesky(A)

println("Matrix A: ")
display(A)

println("\nThe solution to the cholesky factorization")
display(juliaImplementation.L)

A = CuArray(A)

CholesktFactorization(A)
println("\nOur solution to the cholesky factorization")
display(A)

end
export CholeskyTile