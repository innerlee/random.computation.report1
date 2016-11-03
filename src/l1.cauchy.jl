
"""
    Solve L1 regression by sketching.
    Input:
        Ab: concate of A and b
        sk: cauchy / exponential sketching function
        t : num of cols in the gaussian sketching
        r : num of rows in the leverage sketching
    Output:
        f : optimal objective value
        x : optimal point
        t : (time_p, time_leverage, time_tiny_problem)
"""
function l1_sketch_solver(Ab, sk, t, r)
    n, d = size(Ab)

    # cauchy / exponential sketching
    tic()
        skSA = sk(Ab)
        _, R = qr(skSA)

        # gaussian sketching and p
        G = randn(d, t) / sqrt(t)
        p = sum(abs(Ab * (inv(R) * G)), 2)
        p = p / sum(p)
    time_p = toq()

    # leverage score, sample r rows
    tic()
        cols = (1:n)[rand(n) .<= r * p]
        c = length(cols)
        Ω    = sparse(1:c, cols, 1, c, n)
        D    = spdiagm(1 ./ min(1, p[cols] * r))
        # D    = spzeros(c, c)
        # for j = 1:c
        #     D[j, j] = 1 / min(1, p[cols[j]] * r)
        # end
        SAb = D * Ω * Ab
    time_leverage = toq()

    # solve the tiny problem
    tic()
        _, x = l1_solver(SAb)
    time_tiny_problem = toq()

    # real cost
    f = sum(abs(Ab * x)) / n

    f, x, (time_p, time_leverage, time_tiny_problem)
end

"""
    cauchy sketching
"""
cauchy(A, r) = rand(Cauchy(), r, size(A, 1)) * A

"""
    exponential sketching
"""
function exponential(A, r)
    n, d = size(A)
    S = sparse(rand(1:r, n), 1:n, rand([1,-1], n), r, n)
    D = spdiagm(rand(Exponential(), n))
    S * D * Ab
end

"""
    cauchy benchmark
"""
function cauchy_bench(Ab, r_cauchy, r_gauss, r_leverage; repeat=1, verbose=true)

    results = []
    for i =1:repeat
        fs(A) = cauchy(A, r_cauchy)
        res = l1_sketch_solver(Ab, fs, r_gauss, r_leverage)
        push!(results, res)
    end

    f                 = [r[1]    for r in results]
    x                 = [r[2]    for r in results]
    time_p            = [r[3][1] for r in results]
    time_leverage     = [r[3][2] for r in results]
    time_tiny_problem = [r[3][3] for r in results]

    if verbose
        println("\n== cauchy: slove by sketching")
        println("repeat = ", repeat)
        println("r_cauchy = ", r_cauchy)
        println("r_gauss = ", r_gauss)
        println("r_leverage = ", r_leverage)
        println("mean time spent: ", mean(time_p + time_leverage + time_tiny_problem))
        println("    compute p: ", mean(time_p))
        println("    leverage: ", mean(time_leverage))
        println("    tiny problem: ", mean(time_tiny_problem))
        println("min/max/median/mean/std cost: ", round([
            minimum(f), maximum(f), median(f), mean(f), std(f)
            ], 4))
        println("min cost = ", minimum(f))
        println("x = ", x[indmin(f)])
    end

    minimum(f)
end

"""
    exponential benchmark
"""
function exponential_bench(Ab, r_exponential, r_gauss, r_leverage; repeat=1, verbose=true)

    results = []
    for i =1:repeat
        fs(A) = exponential(A, r_exponential)
        res = l1_sketch_solver(Ab, fs, r_gauss, r_leverage)
        push!(results, res)
    end

    f                 = [r[1]    for r in results]
    x                 = [r[2]    for r in results]
    time_p            = [r[3][1] for r in results]
    time_leverage     = [r[3][2] for r in results]
    time_tiny_problem = [r[3][3] for r in results]

    if verbose
        println("\n== exponential: slove by sketching")
        println("repeat = ", repeat)
        println("r_exponential = ", r_exponential)
        println("r_gauss = ", r_gauss)
        println("r_leverage = ", r_leverage)
        println("mean time spent: ", mean(time_p + time_leverage + time_tiny_problem))
        println("    compute p: ", mean(time_p))
        println("    leverage: ", mean(time_leverage))
        println("    tiny problem: ", mean(time_tiny_problem))
        println("min/max/median/mean/std cost: ", round([
            minimum(f), maximum(f), median(f), mean(f), std(f)
            ], 4))
        println("min cost = ", minimum(f))
        println("x = ", x[indmin(f)])
    end

    minimum(f)
end
