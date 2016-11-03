
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
        D    = spzeros(c, c)
        for j = 1:c
            D[j, j] = 1 / min(1, p[cols[j]] * r)
        end
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
    cauchy benchmark
"""
function cauchy_bench(Ab, r_cauchy, r_gauss, r_leverage; repeat=1, verbose=true)
    fs(A) = cauchy(A, r_cauchy)
    f, x, (time_p, time_leverage, time_tiny_problem) =
        l1_sketch_solver(Ab, fs, r_gauss, r_leverage)

    if verbose
        println("\n== cauchy: slove by sketching")
        println("time spent in:")
        println("    compute p: ", time_p)
        println("    leverage: ", time_leverage)
        println("    tiny problem: ", time_tiny_problem)
        println("cost: ", f)
        println("x = ", x)
    end

    f, x, (time_p, time_leverage, time_tiny_problem)
end
