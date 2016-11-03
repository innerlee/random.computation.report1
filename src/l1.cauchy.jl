
"""
    Solve L1 regression by sketching.
    Input:
        Ab: concate of A and b
        p : p vector used in the pipeline
        r : num of rows in the leverage step
    Output:
        f : optimal objective value
        x : optimal point
        t : (time_leverage, time_tiny_problem)
"""
function l1_sketch_solver(Ab, p, r)
    n, d = size(Ab)

    # leverage score, sample r rows
    tic()
    cols = sample(1:n, WeightVec(p), r)
    Ω    = sparse(1:r, cols, 1, r, n)
    D    = spzeros(r, r)
    for j = 1:r
        D[j, j] = 1 / min(1, p[cols[j]] * r)
    end
    SAb = D * Ω * Ab
    time_leverage = toq()

    # solve the tiny problem
    tic()
    _, x = l1_solver(SAb)
    time_tiny_prob = toq()

    # real cost
    f = sum(abs(Ab * x)) / n

    f, x, (time_leverage, time_tiny_problem)
end

"""
    get p vector
"""
function cauchy_p(Ab, r, t)
    n, d = size(Ab)

    SA =
    Q, R = qr(SA)
    G = randn(d, t) / sqrt(t)
    p = sum(abs(A * (inv(R) * G)), 2)

    p / sum(p)
end

"""
    cauchy benchmark
"""
function cauchy_bench(Ab, r_cauchy, r_gauss, r_leverage; repeat=1, verbose=true)
    tic()
    p = cauchy_p(Ab, r_cauchy, r_gauss)
    time_cauchy = toq()
    f, x, (time_leverage, time_tiny_problem) = l1_sketch_solver(Ab, p, r_leverage)

    if verbose
        println("\n== cauchy: slove by skeching")
        println("time spent in:")
        println("    cauchy: ", time_cauchy)
        println("    leverage: ", time_leverage)
        println("    tiny problem: ", time_tiny_problem)
        println("cost: ", f)
        println("x = ", x)
    end

    f, x, t
end
