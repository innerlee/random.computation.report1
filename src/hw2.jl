# testing code for hw2
using JLD
using JuMP
using StatsBase
using Distributions

include("l1.baseline.jl")
include("l1.cauchy.jl")
include("l1.exponential.jl")

# config
MAX_SAMPLE = 10000

# load data
data = load("../output/data.jld");
A    = data["A"]
b    = data["b"]

# normalize data
A  = (A .- mean(A, 1)) ./ std(A, 1)
b  = (b .- mean(b, 1)) ./ std(b, 1)
Ab = hcat(A, b)

# truncate data
MAX_SAMPLE = min(MAX_SAMPLE, size(A, 1))
Ab         = Ab[1:MAX_SAMPLE, :]

n, d = size(Ab)

rel_err(base, comp) = println("relative err: $(round((comp / base - 1) * 100, 2))%")

# baseline
cost0, _, _ = baseline_bench(Ab)

# cauchy
r_cauchy   = 64
r_gauss    = 16
r_leverage = 100
cost_c = cauchy_bench(Ab, r_cauchy, r_gauss, r_leverage, repeat=100)
rel_err(cost0, cost_c)

# # exponential
r_exp      = 64
r_gauss    = 16
r_leverage = 100
cost_e = exponential_bench(Ab, r_exp, r_gauss, r_leverage, repeat=100)
rel_err(cost0, cost_e)
