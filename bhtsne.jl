module BHTsne

using Statistics: mean
using LinearAlgebra: eigen, Symmetric, transpose

# Constants
BH_TSNE_BIN_PATH = ifelse(Sys.iswindows(),
    joinpath(dirname(@__FILE__), "cpp/bh_tsne.exe"),
    joinpath(dirname(@__FILE__), "cpp/bh_tsne"))

if !isfile(BH_TSNE_BIN_PATH)
    println("Unable to find the bh_tsne binary in the same
    directory as this script, trying to compile...")
    cd(dirname(BH_TSNE_BIN_PATH)) do
        run(`g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O3`)
    end
end

cI32(x) = convert(Int32, x)
rI32(x) = read(x, Int32)
rF64(x) = read(x, Float64)

function pca(X::AbstractMatrix, ndims::Integer = 50)
    (n, d) = size(X)
    (d <= ndims) && return X
    Y = X .- mean(X, dims = 1)
    C = Symmetric((Y' * Y) ./ (n - 1))
    # take eigvects for top ndims largest eigvals
    Ceig = eigen(C, (d - ndims + 1):d)
    return Y * reverse(Ceig.vectors, dims = 2)
end

function bh_tsne(X, no_dims::Int = 2, initial_dims::Int = 50,
    perplexity::Float64 = 50.0, theta::Float64 = 0.5, randseed::Int = -1,
    max_iter::Int = 1000; verbose = true, use_pca = true)

    if use_pca
        samples = pca(X, initial_dims)
    end
    # Assume that the dimensionality of the first sample is representative
    # for the whole batch
    sample_count, sample_dim = size(samples)

    # Create a temporary directory for the binary files
    mktempdir() do temp_dir
        open(joinpath(temp_dir, "data.dat"), "w") do data_file
            write(data_file,
                cI32(sample_count),
                cI32(sample_dim),
                theta,
                perplexity,
                cI32(no_dims),
                cI32(max_iter))
            # The executable assumes a row-major ordering for arrays
            for i in 1:size(samples', 2)
                write(data_file, samples'[:, i])
            end
            if randseed != -1
                write(data_file, cI32(randseed))
            end
            close(data_file)
        end

        cd(temp_dir) do
            if verbose
                try
                    run(`$BH_TSNE_BIN_PATH`)
                catch excp
                    println(excp)
                    error("Call to bh_tsne exited with non-zero code exit status,
                          please refer to the bh_tsne output for further detail.")
                end
            else
                try
                    run(pipeline(`$BH_TSNE_BIN_PATH`,
                        stdout = nothing,
                        stderr = nothing))
                catch excp
                    println(excp)
                    error("Call to bh_tsne exited with a non-zero return code exit status,
                          please enable verbose mode and refer to the bh_tsne output for further details")
                end
            end
        end

        open(joinpath(temp_dir, "result.dat"), "r") do output_file
            result_samples = rI32(output_file)
            result_dims = rI32(output_file)
            # Allocate an uninitialized array for results
            results = Array{Float64,2}(undef, result_dims, result_samples)
            read!(output_file, results)
            results = transpose(results)
            ret = [(rI32(output_file), results[i, :]) for i in 1:result_samples]
            close(output_file)
            ret = sort(ret)
            for i in 1:result_samples
                results[i, :] .= ret[i][2]
            end
            return results
        end
    end
end

end  # module BHTsne
