# testing code for hw2
using JLD
using JuMP
using StatsBase
using Distributions

include("l1.baseline.jl")
include("l1.cauchy.jl")
include("l1.exponential.jl")

# config
MAX_SAMPLE = 2000

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
seed_data  = Ab[1:10, :]

n, d = size(Ab)

# # baseline
# baseline_bench(seed_data, verbose=false)
# baseline_bench(Ab)

# cauchy
r_cauchy   = 64
r_gauss    = 16
r_leverage = 100
# cauchy_bench(seed_data, verbose=false)
cauchy_bench(Ab, r_cauchy, r_gauss, r_leverage)

# # exponential
# exponential_bench(seed_data, verbose=false)
# exponential_bench(Ab)
