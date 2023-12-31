module GPUUtils
using CUDA
using KernelAbstractions


@kernel function isequal_kernel!(A, B, epsilon)
    i, j = @index(Global, NTuple)
    if abs(A[i, j] - B[i, j]) > epsilon
        # we say that the array is not equal
        A[i, j] = 0
    else
        A[i, j] = 1
    end
end
"""
Given two CUDA Arrays A and B, compute if they are equal or not
"""
function isEqual(A, B, epsilon=0.01)
    backend = KernelAbstractions.get_backend(A)
    kernel! = isequal_kernel!(backend)

    n, m = size(A)
    C = CuArray(ones(n, m))
    copyto!(C, A)

    kernel!(C, B, epsilon, ndrange=size(A))
    KernelAbstractions.synchronize(backend)

    return findmin(C)[1] == 1
end

export GPUUtils
end