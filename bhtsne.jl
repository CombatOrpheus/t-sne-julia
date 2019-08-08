module BHTsne

using ArgParse
using Statistics: mean
using LinearAlgebra: eigen, Symmetric

# Constants
BH_TSNE_BIN_PATH = ifelse(
    Sys.iswindows(),
    joinpath(dirname(@__FILE__), "cpp/bh_tsne.exe"),
    joinpath(dirname(@__FILE__), "cpp/bh_tsne")
)

if !isfile(BH_TSNE_BIN_PATH)
    println("Unable to find the bh_tsne binary in the same
    directory as this script, trying to compile...")
    cd(dirname(BH_TSNE_BIN_PATH)) do
        run(`g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O3`)
    end
end

#=
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-d","--no_dims"
            arg_type = Int64
        "-p","--perplexity"
            arg_type = Float64
            default = 50
        "-t","--theta"
            arg_type = Float64
            default = 0.5
        "-r","--randseed"
            arg_type = Int64
            default = -1
        "-n","--initial_dims"
            arg_type = Int64
            default = 50
        "-v","--verbose"
            action = :store_true
        "-i","--input"
            arg_type = ASCIIString
            default = STDIN
        "-o","--output"
            arg_type = ASCIIString
            default = STDOUT
    end

    parse_args(s)
end
=#

cI32(x) = convert(Int32, x)

function pca(X::AbstractMatrix, ndims::Integer = 50)
    (n, d) = size(X)
    (d <= ndims) && return X
    Y = X .- mean(X, dims = 1)
    C = Symmetric((Y' * Y) ./ (n - 1))
    Ceig = eigen(C, (d-ndims+1):d) # take eigvects for top ndims largest eigvals
    return Y * reverse(Ceig.vectors, dims = 2)
end

function bh_tsne(X, no_dims::Int = 2, initial_dims::Int = 50,
    perplexity::Float64 = 50.0, theta::Float64 = 0.5, randseed::Int = -1,
    max_iter::Int = 1000; verbose = true, use_pca = true)

    if use_pca
        samples = pca(samples, initial_dims)
    end
    # Assume that the dimensionality of the first sample is representative for
    # the whole batch
    sample_count, sample_dim = size(X)

    mktempdir() do temp_dir

        open(joinpath(temp_dir, "data.dat"), "w") do data_file
            write(
                data_file,
                cI32(sample_count),
                cI32(sample_dim),
                theta,
                perplexity,
                cI32(no_dims),
                cI32(max_iter)
            )
            for i in eachrow(X')
                write(data_file, i)
            end
            if randseed != -1
                write(data_file, cI32(randseed))
            end
            close(data_file)
        end

        cd(temp_dir) do
            if verbose
                try
                    run(pipeline(
                        `$BH_TSNE_BIN_PATH`,
                        stdout = "out.txt",
                        stderr = "errs.txt"
                    ))
                catch excp
                    println(excp)
                    error("ERROR: Call to bh_tsne exited with non-zero code exit status,
                          please refer to the bh_tsne output for further detail.")
                end
            else
                try
                    run(pipeline(`$BH_TSNE_path`, stdout = DevNull))
                catch excp
                    println(excp)
                    error("ERROR: Call to bh_tsne exited with a non-zero return code exit status,
                          please enable verbose mode and refer to the bh_tsne output for further details")
                end
            end
        end

        open(joinpath(temp_dir, "result.dat"), "r") do output_file
            result_samples = read(output_file, Int32)
            result_dims = read(output_file, Int32)
            results = [(read(output_file, Int), read(output_file, Int)) for _ in 1:result_samples]
            results = [(read(output_file, Int), e) for e in results]
            sort!(results)
            ret = Array{Float64,2}(result_dims, result_samples)
            for i in 1:result_samples
                ret[:, i] = collect(results[i][2])
            end
            return ret'
        end
    end

end



end  # module BHTsne
