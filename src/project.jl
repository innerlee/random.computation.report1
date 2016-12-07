# codes for project
using JLD
using DataFrames

# data preprocess
include("preprocess_movie.jl")

"""
    test_svd()

Benchmark svd on the full matrix.
"""
function test_svd()
    # 26744×138493 sparse matrix with 20000263 Float64 nonzero entries
    data = full(load_movielens())
    tic()
    u, s, v = svd(data)
    toc()
    save("output/svd_movie.jld", "u", u, "s", s, "v", v)
    (u, s, v)
end
