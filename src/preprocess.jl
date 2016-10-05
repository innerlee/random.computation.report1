using JLD
using StatsBase

function save_valid_data()
    # 206s to load "train.csv"
    # 13765201×24 data
    # header:
    #   Id, minutes_past, radardist_km, Ref, Ref_5x5_10th, Ref_5x5_50th,
    #   Ref_5x5_90th, RefComposite, RefComposite_5x5_10th, RefComposite_5x5_50th,
    #   RefComposite_5x5_90th, RhoHV, RhoHV_5x5_10th, RhoHV_5x5_50th,
    #   RhoHV_5x5_90th, Zdr, Zdr_5x5_10th, Zdr_5x5_50th, Zdr_5x5_90th, Kdp,
    #   Kdp_5x5_10th, Kdp_5x5_50th, Kdp_5x5_90th, Expected
    @time data, title = readcsv("../data/train.csv", header=true)

    # 204s
    @time missing = map(sum, data[i, :] .== "" for i = 1:size(data, 1))

    # 2769088×21 valid features (no missing value)
    A = map(Float64, data[missing .== 0, 3:23])
    b = map(Float64, data[missing .== 0, 24])

    save("../output/valid.data.jld", "A", A, "b", b)
end

function save_valid_data_no_outlier()
    d = load("../output/valid.data.jld")
    p = percentile(d["b"], 99)
    fil = d["b"] .< p
    println("99-th percentile of b = ", p)
    println("there are $(sum(fil)) points")

    save("../output/data.jld", "A", d["A"][fil, :], "b", d["b"][fil])
end
