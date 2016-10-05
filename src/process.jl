using JLD

function baseline()
    # directly apply on all valid data A_0, b_0
    d = load("../output/valid.data.jld")
    A = d["A"]
    b = d["b"]
    @time x = (A' * A) \ (A' * b)
    err = norm(A * x - b)
    println("directly computation, original data, error = ", err,
        ", average error = ", err / sqrt(length(b)),
        ", std(b)) = ", std(b))

    # after filtered outliers. use data A, b
    d = load("../output/data.jld")
    A = d["A"]
    b = d["b"]
    @time x = (A' * A) \ (A' * b)
    err = norm(A * x - b)
    println("directly computation, original data, error = ", err,
        ", average error = ", err / sqrt(length(b)),
        ", std(b)) = ", std(b))
end
