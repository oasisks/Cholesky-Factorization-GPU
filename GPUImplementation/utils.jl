module utils
using KernelAbstractions
using CUDA

@kernel function getTile(i::Int64, j::Int64, M::Matrix, tile_size::Int64)
    I = @index(Global)
    # assume square matrices
    tile_i = tile_size * (I + i) - (tile_size - 1)
    tile_i_e = tile_size * (I + i)
    
    tile_j = tile_size * j - (tile_size - 1)
    tile_j_e = tile_size * j

    return M[tile_i: tile_i_e, tile_j: tile_j_e]
end

function setTile!(i::Int64, j::Int64, M::Matrix, tile_size::Int64, value::Matrix)
    I = @index(Global)
    # assume square matrices
    tile_i = tile_size * (I + i) - (tile_size - 1)
    tile_i_e = tile_size * (I + i)
    
    tile_j = tile_size * j - (tile_size - 1)
    tile_j_e = tile_size * j

    M[tile_i: tile_i_e, tile_j: tile_j_e] .= value
end

export utils
end