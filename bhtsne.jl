module BHTsne

using ArgParse
using Statistics: mean
using LinearAlgebra: eigen

# Constants
BH_TSNE_BIN_PATH = ifelse(Sys.iswindows(),
                          joinpath(dirname(@__FILE__), "cpp/bh_tsne.exe"),
                          joinpath(dirname(@__FILE__), "cpp/bh_tsne"))

if !isfile(BH_TSNE_BIN_PATH)
    println("Unable to find the bh_tsne binary in the same
    directory as this script, trying to compile...")
    cd(dirname(BH_TSNE_BIN_PATH)) do
        run(`g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2`)
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

# Starting here, most functions are in Python

function bh_tsne(samples, no_dims=2, initial_dims=50,
                      perplexity=50, theta=0.5, randseed=-1, verbose=true)

    # if use_pca
        samples .-= mean(samples, dims=1)
        cov_x = samples' * samples
        eig_val, eig_vec = eigen(cov_x)

        # Sort the eigen-values in the descending order
        eig_vec = eig_vec[:, sortperm(eig_val, rev=true)]

        if initial_dims > length(eig_vec)
        initial_dims = length(eig_vec)
        # end

        # truncate the eigen-vectors matrix to keep the most vectors
        # eig_vec = real.(eig_vec[:, :initial_dims])
        samples = samples * eig_vec
    end

    # Assume that the dimensionality of the first sample is representative for
    # the whole batch
    sample_count, sample_dim = size(samples)

    mktempdir() do temp_dir

        open(joinpath(temp_dir, "data.dat"), "w") do data_file
        #Python
            parameters = [sample_count,sample_dim, theta, perplexity,no_dims]
            map(x -> write(data_file, x), parameters)
            nrow, ncol = size(samples)
            fmt = repeat("d", ncol)
            for i in 1:nrow
                write(data_file, samples[i, :])
            end
            if randseed != -1
                write(data_file, randseed)
            end
        end

        cd(temp_dir) do
            if verbose
                try
                    run(pipeline(`$BH_TSNE_BIN_PATH`, stdout=STDERR))
                catch excp
                # warn(excp)
                error("ERROR: Call to bh_tsne exited with non-zero code exit status,
                      please refer to the bh_tsne output for further detail.")
                end
            else
                try
                    run(pipeline(`$BH_TSNE_path`, stdout=DevNull))
                catch excp
                    # warn(excp)
                    error("ERROR: Call to bh_tsne exited with a non-zero return code exit status,
                          please enable verbose mode and refer to the bh_tsne output for further details")
                end
            end
        end

        open(joinpath(temp_dir, "result.dat"), "r") do output_file
            result_samples = read(output_file, Int)
            result_dims = read(output_file, Int)
            results = [(read(output_file, Int), read(output_file, Int)) for _ in 1:result_samples]
            results = [(read(output_file, Int), e) for e in results]
            sort!(results)
            ret = Array{Float64, 2}(result_dims, result_samples)
            for i in 1:result_samples
                ret[:, i] = collect(results[i][2])
            end
            return ret'
        end
    end

end



end  # module BHTsne
