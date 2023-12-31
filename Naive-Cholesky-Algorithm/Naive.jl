module Naive

"""
For locality, we did it by column because julia is column based caching
"""
function NaiveCholesky(matrix::Matrix)
    n = size(matrix, 1)
    L = zeros(ComplexF64, n, n)

    for j = 1:n
        sum = 0
        for k = 1:j
            sum += L[j, k] * conj(L[j, k])
        end
        L[j, j] = sqrt(matrix[j, j] - sum)

        for i = j + 1:n 
            sum = 0
            for k = 1:j 
                sum += L[i, k] * conj(L[j, k])
            end
            L[i, j] = (1.0 / L[j, j] * (matrix[i, j] - sum))
        end 
    end
    return (L = L, U = adjoint(L))
end

export NaiveCholesky
end