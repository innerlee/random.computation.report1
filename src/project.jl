# codes for project
using JLD
using HDF5
using DataFrames

# data preprocess
include("preprocess_movie.jl")

"""
    test_svd()

Benchmark svd on the full matrix.
And save them to files.
"""
function test_svd()
    # 26744×138493 sparse matrix with 20000263 Float64 nonzero entries
    data = full(load_movielens())
    tic()
    # 21403s
    u, s, v = svd(data)
    toc()
    # 5.4G
    h5write("output/svd_u.h5", "u", u)
    # 212K
    h5write("output/svd_s.h5", "s", s)
    # 28G
    h5write("output/svd_v.h5", "v", v)
    (u, s, v)
end

"""
    load_svd()

Load the exact svd result.
"""
function load_svd()
    u = h5read("output/svd_u.h5", "u")
    s = h5read("output/svd_s.h5", "s")
    v = h5read("output/svd_v.h5", "v")
    (u, s, v)
end

"""
    bench_k(krange=10:10:100)

Find a good rank k for approximation.

Result:
energy = 16453.6
k   err     time(s)
10  13408.6 9.7
20  13003.8 19.1
30  12754.6 25.7
40  12571.0 35.1
50  12419.6 45.1
60  12287.5 57.5
70  12169.3 73.9
80  12060.1 81.0
90  11958.8 106.7
100 11863.5 131.8
200 11101.7 241.4
300 10506.1 559.8
"""
function bench_k(krange=10:10:100)
    u0, s0, v0 = load_svd()
    energy = vecnorm(s0)
    A = load_movielens()
    for k in krange
        println("> use rank k = ", k)
        tic()
        # use svds
        ss = svds(A, nsv=k)
        toc()
        # compute diff
        dif = vecnorm(s0[k+1:end])
        println("||A-A_k||_F = ", round(dif,1))
    end
end
