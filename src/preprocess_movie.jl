# preprocess MovieLens 20M dataset

"""
    movie_process()

Convert the raw rating data to a sparse matrix.
- Rows are movies and columns are users.
- Elements are ratings.
"""
function movie_process()
    # read raw data 20000263×4. [userId │ movieId │ rating │ timestamp]
    df = readtable("data/ml-20m/ratings.csv")
    # user ids are continuous.
    assert(length(unique(df[:userId])) == maximum(df[:userId]))
    # compress the movie ids. 26744 movies with maximum id 131262.
    mids = sort(unique(df[:movieId]))
    mid_compress = Dict()
    mid_recover = Dict()
    for (i, mid) in enumerate(mids)
        mid_compress[mid] = i
        mid_recover[i] = mid
    end
    # build sparse matrix. row: movie, col: user, elem: rating
    # 26744×138493 sparse matrix with 20000263 Float64 nonzero entries
    sparse([mid_compress[i] for i in df[:movieId]], df[:userId], df[:rating])
end

"""
    save_process_data()

Save the ratings sparse matrix to a file.
"""
save_process_data() = save("output/movielens20m.jld", "rating", movie_process())

"""
    load_movielens()

Load the rating sparse matrix.
"""
load_movielens() = load("output/movielens20m.jld")["rating"]
