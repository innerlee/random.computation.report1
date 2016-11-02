
"""
    Solve L1 regression by Cauchy.
    Input:
        Ab: concate of A and b
    Output:
        f : optimal objective value
        x : optimal point
"""
function cauchy_solver(Ab)
    M, N = size(Ab)

end

"""
    cauchy benchmark
"""
function cauchy_bench(Ab; verbose=true)
    tic()
    f, x = cauchy_solver(Ab)
    t = toq()

    if verbose
        println("\n== cauchy: slove by skeching ")
        println("time used: ", t)
        println("objective value: ", f)
        println("x = ", x)
    end

    f, x, t
end
