# codes for project
using JLD
using HDF5
using DataFrames

"""
    movie_process()

Preprocess MovieLens 20M dataset
Convert the raw rating data to a sparse matrix.
- Rows are movies and columns are users.
- Elements are ratings.
"""
function movie_process()
    # read raw data 20000263×4. [userId │ movieId │ rating │ timestamp]
    df = readtable("data/ml-20m/ratings.csv")
    # user ids are continuous.
    assert(length(unique(df[:userId])) == maximum(df[:userId]))
    # compress the movie ids. 26744 movies with maximum id 131262.
    mids = sort(unique(df[:movieId]))
    mid_compress = Dict()
    mid_recover = Dict()
    for (i, mid) in enumerate(mids)
        mid_compress[mid] = i
        mid_recover[i] = mid
    end
    # build sparse matrix. row: movie, col: user, elem: rating
    # 26744×138493 sparse matrix with 20000263 Float64 nonzero entries
    sparse([mid_compress[i] for i in df[:movieId]], df[:userId], df[:rating])
end

"""
    save_process_data()

Save the ratings sparse matrix to a file.
"""
save_process_data() = save("output/movielens20m.jld", "rating", movie_process())

"""
    load_movielens()

Load the rating sparse matrix.
"""
load_movielens() = load("output/movielens20m.jld")["rating"]

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
k   F err   2 err   time(s)
100 11863.5  131.8
200 11101.7  241.4
300 10506.1  559.8
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
        dif2 = s0[k+1]
        println("||A-A_k||_F = ", round(dif,1))
        println("||A-A_k||_2 = ", round(dif2,1))
    end
end

"""
    sketch_frob(A, fA; k=128, t=256)

Arguments
    A : sparse A
    fA: full A
    k : rank
    t : sketch dim

Returns
    ((uk, sk, vk), (time_sketch, time_tiny_problem))

Use count sketch to reduce the column space.
"""
function sketch_frob(A, fA; k=128, t=256)
    # count sketch, reduce columns
    tic()
        # 138493 × 26744
        n = size(A, 1)
        # s × 138493
        S = sparse(rand(1:t, n), 1:n, rand([1,-1], n))
        # s × 26744
        SA = full(S * A)
    time_sketch = toq()
    tic()
        Q, R = qr(SA')
        # full A is faster in multiplication
        AQ = fA * Q
        u, s, v = svd(AQ)
        uk = u[:, 1:k]
        sk = s[1:k]
        vk = v[:, 1:k]
        usv = (uk * spdiagm(sk)) * (vk' * Q')
    time_tiny_problem = toq()
    err = vecnorm(fA-usv)
    ((uk, sk, vk), err, (time_sketch, time_tiny_problem))
end

"""
    bench_sketch_frob(repeat=1, tt=[256])

Arguments
    repeat: how many repeats for sketch
    tt    : collection of columns that sketch reduces to

sketch benchmark for frobenius norm.
"""
function bench_sketch_frob(repeat=1, tt=[256])
    # 138493 × 26744
    A = load_movielens()'
    fA = full(A)
    t0 = 111.4
    e0 = 11622.2
    for t in tt
        println("> will reduce A with size $(size(A)) to $t rows")
        results = []
        for i = 1:repeat
            r = sketch_frob(A, fA, t=t)
            push!(results, r)
        end

        # usv               = [r[1]    for r in results]
        err               = [r[2]    for r in results]
        time_sketch       = [r[3][1] for r in results]
        time_tiny_problem = [r[3][2] for r in results]

        println("repeat = ", repeat)
        println("mean time spent = ", mean(time_sketch + time_tiny_problem))
        println("          ratio = ", mean(time_sketch + time_tiny_problem) / t0)
        println("    sketch: ", mean(time_sketch))
        println("    tiny problem: ", mean(time_tiny_problem))
        println("min/max/median/mean/std err (frob norm): ", round([
            minimum(err), maximum(err), median(err), mean(err), std(err)
            ], 4))
        println("min err (frob norm) = ", minimum(err))
        println("              ratio = ", minimum(err) / e0)
    end
end

"""
    maxeig(A)

Compute maximum eig of A by power method.
"""
function maxeig(A; iter=81)
    v = normalize(randn(size(A, 2)))
    for i = 1:iter
        v = normalize(A*v)
        v = normalize(A'*v)
    end
    norm(A * v)
end

"""
    sub_power(fA; k=128, q=4)

Argumeents
    fA  : full matrix A
    k   : rank
    q   : power

Low rank approx. w.r.t. operator norm by subspace power method.
Error is evaluated by power method.
"""
function sub_power(fA, fAt; k=128, q=4)
    tic()
        n = size(fA, 2)
        G = randn(n, k)
        Y = fA * G
        Y, _ = qr(Y)
        for i = 1:q
            Y = fAt * Y
            Y = fA * Y
            Y, _ = qr(Y)
        end
        Z, _ = qr(Y)
        # the approximation
        Ak = Z * (Z' * fA)
    time_power = toq()
    # compute error using power method
    err = maxeig(fA - Ak)
    (Z, err, time_power)
end

"""
    bench_sub_power(repeat=1, qq=[256])

Arguments
    repeat: how many repeats for sketch
    qq    : collection of numbers that power took

sketch benchmark for the operator norm.
"""
function bench_sub_power(repeat=1, qq=[4])
    # 138493 × 26744, did a transpose for this shape.
    fA = full(load_movielens()')
    fAt = fA'
    t0 = 111.4
    e0 = 432.6
    for q in qq
        println("> will power $q times")
        results = []
        for i = 1:repeat
            r = sub_power(fA, fAt, q=q)
            push!(results, r)
        end

        zzz               = [r[1] for r in results]
        err               = [r[2] for r in results]
        time_power        = [r[3] for r in results]

        println("repeat = ", repeat)
        println("mean time spent = ", mean(time_power))
        println("          ratio = ", mean(time_power) / t0)
        println("min/max/median/mean/std err (op norm): ", round([
            minimum(err), maximum(err), median(err), mean(err), std(err)
            ], 4))
        println("min err (op norm) = ", minimum(err))
        println("            ratio = ", minimum(err) / e0)
    end
end

## config
REPEAT = 100
TT     = [256, 512, 1024]
REPEAT2 = 16
QQ     = [4, 8, 16]

# main
bench_sketch_frob(REPEAT, TT)
bench_sub_power(REPEAT2, QQ)
