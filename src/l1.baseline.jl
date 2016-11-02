
"""
    Solve L1 regression by LP.
    Input:
        Ab: concate of A and b
    Output:
        f : optimal objective value
        x : optimal point
"""
function l1_solver(Ab)
    M, N = size(Ab)

    # baseline, LP solver by JuMP
    m = Model()
    @variable(m, t[1:M] >= 0 )
    @variable(m, x[1:N])

    @objective(m, Min, sum(t))

    @constraint(m, Ab * x .<= t )
    @constraint(m, -t .<= Ab * x )
    @constraint(m, x[N] == -1 )

    solve(m) == :Optimal || error("no solution found!")

    (getobjectivevalue(m) / M, getvalue(x)[1:N-1])
end

"""
    baseline benchmark
"""
function baseline_bench(Ab; verbose=true)
    tic()
    f, x = l1_solver(Ab)
    t = toq()

    if verbose
        println("\n== baseline: slove by LP ")
        println("time used: ", t)
        println("objective value: ", f)
        println("x = ", x)
    end

    f, x, t
end
