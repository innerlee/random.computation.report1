using JLD

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

    # Gaussian
    ck = floor(Int, (d + log2(1 / δ)) / ɛ^2)
    println("Gaussian, computed k = ", ck)

    println("actual use k = ", k)

    # result format for each try (x, err, time_prepare, time_apply)
    results = []
    # try multiple times
    for i = 1:ntry
        verbose && print(".")
        tic()
        SAb = vcat([randn(1, n) * Ab / sqrt(k) for i=1:k]...)
        time_prepare = toq()
        tic()
            x = simple(SAb)
        time_apply = toq()
        err = loss(Ab, x)
        push!(results, (x, err, time_prepare, time_apply))
    end

    results
end

# config
ntry = 100
k = 100

# main
r = go(ntry, k)
xs = [r[i][1] for i = 1:length(r)]
errs = [r[i][2] for i = 1:length(r)]
time_prepares = [r[i][3] for i = 1:length(r)]
time_applys = [r[i][4] for i = 1:length(r)]

println("repeat $(length(r)) times")
println("k = ", k)
println("mean prepare time = ", mean(time_prepares))
println("mean apply time = ", mean(time_applys))
println("mean error = ", mean(errs))
println("x = ", xs[indmin(errs)])
println("min error = ", minimum(errs))