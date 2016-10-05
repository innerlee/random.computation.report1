using JLD
using Hadamard

function go(ntry=1, k=100; verbose=true)
    # load data
    data = load("../output/data.jld")
    A = data["A"]
    b = data["b"]
    Ab = hcat(A, b)

    # configs
    n, d = size(A)
    ɛ = 0.1
    δ = 0.1

    # Simple
    simple(A, b) = (A' * A) \ (A' * b)
    simple(Ab) = simple(Ab[:, 1:end-1], Ab[:, end])
    loss(A, b, x) = norm(A * x - b)
    loss(Ab, x) = loss(Ab[:, 1:end-1], Ab[:, end], x)

    # PHD
    ck = floor(Int, log2(d) * (sqrt(d) + sqrt(log2(n)))^2 / ɛ^2)
    verbose && println("PHD, computed k = ", ck)

    verbose && println("actual use k = ", k)

    # result format for each try (x, err, time_prepare, time_apply)
    results = []
    # try multiple times
    for i = 1:ntry
        verbose && print(".")
        tic()
        # sample k rows
        P = []
        while length(P) < k
            P = unique(rand(1:n, k))
        end
        # D
        DAb = zeros(nextpow2(n), d + 1)
        DAb[1:n, :] = rand([1,-1], n) .* Ab
        # PH
        PHDAb = fwht(DAb, 1)[P, :]
        time_prepare = toq()
        tic()
            x = simple(PHDAb)
        time_apply = toq()
        err = loss(Ab, x)
        push!(results, (x, err, time_prepare, time_apply))
    end
    println()

    results
end

# config
length(ARGS) != 2 && error("args: ntry, k")
ntry = eval(parse(ARGS[1]))
k = eval(parse(ARGS[2]))

# main
go(1, 10, verbose=false)
r = go(ntry, k)
xs = [r[i][1] for i = 1:length(r)]
errs = [r[i][2] for i = 1:length(r)]
time_prepares = [r[i][3] for i = 1:length(r)]
time_applys = [r[i][4] for i = 1:length(r)]

println("repeat $(length(r)) times")
println("k = ", k)
println("mean prepare time = ", mean(time_prepares))
println("mean apply time = ", mean(time_applys))
println("min/median/max/std/mean error = ", round(
    [minimum(errs), median(errs), maximum(errs), std(errs), mean(errs)],
    1))
println("x = ", round(xs[indmin(errs)], 3))
