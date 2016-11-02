
"""
    Solve L1 regression by exponential.
    Input:
        Ab: concate of A and b
    Output:
        f : optimal objective value
        x : optimal point
"""
function exponential_solver(Ab)
    M, N = size(Ab)

end

"""
    exponential benchmark
"""
function exponential_bench(Ab; verbose=true)
    tic()
    f, x = exponential_solver(Ab)
    t = toq()

    if verbose
        println("\n== exponential: slove by skeching ")
        println("time used: ", t)
        println("objective value: ", f)
        println("x = ", x)
    end

    f, x, t
end
