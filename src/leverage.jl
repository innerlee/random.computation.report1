using JLD
using StatsBase

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

    # Leverage Score
    ck = floor(Int, 144 * d * log(2 * d / δ) / (ɛ^2))
    verbose && println("Leverage Score, computed k = ", ck)

    verbose && println("use k = ", k)

    # result format for each try (x, err, time_prepare, time_apply)
    results = []
    # try multiple times
    for i = 1:ntry
        verbose && print(".")
        tic()
            # Ab = ZR
            Z, _, _ = svd(Ab)
            p = sum(Z .* Z, 2)[:] / (d + 1)
            # S = DΩ
            Ω = spzeros(k, n)
            D = spzeros(k, k)
            for j = 1:k
                Ω[j, sample(WeightVec(p))] = 1
                D[j, j] = 1 / sqrt(p[j] * k)
            end
            SAb = D * Ω * Ab
        time_prepare = toq()
        tic()
            x = simple(SAb)
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
