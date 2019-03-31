module TSNE

using Printf: @printf
using LinearAlgebra
using Statistics: mean

function Hbeta!(P::AbstractVector{T}, D::AbstractVector{T},
     beta=1.0) where T<:Number
    @inbounds P .= exp.(-D .* beta)
    sumP = sum(P)
    H = log(sumP) + beta * sum(D .* P) / sumP
    @inbounds P ./= sumP
    return H
end

function diagzero!(X::AbstractMatrix{T}) where T<:Number
    n = zero(T)
    for i in size(X, 1)
        X[i, i] = n
    end
end

#TODO: Verify this function
function d2p(D::AbstractMatrix{T}, perplexity=15, tol=14e-4) where T<:Number
    # Initializing variables
    n = size(D, 1)          #   Number of instances
    P = zeros(n, n)         #   Empty probability Matrix
    beta = fill(one(T), n)  #   Empty precision vector
    logU = log(perplexity)  #   Log of perplexity
    Pcol = fill(zero(T), n)

    # Run over all datapoints
    for i in 1:n
        if i % 500 == 0
            @printf("Computed P-values %d of %d datapoints...\n", i, n)
        end

        # Set minimun and maximum values for precision
        betamin = 0.0
        betamax = Inf
        betai = 1.0

        # Compute the Gaussian kernel and entropy for the current precision
        Pi = view(P, :, i)
        Di = view(D, :, i)
        Di[i] = prevfloat(Inf)
        H = Hbeta!(Pi, Di, betai)

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol && tries < 50
            if Hdiff > 0.0
                betamin = betai
                betai = isfinite(betamax) ? (betai + betamax)/2 : betai*2
            else
                betamax = betai
                betai = (betai + betamin)/2
            end

            # Recompute the values
            H = Hbeta!(Pi, Di, betai)
            Hdiff = H -logU
            tries += 1
        end
        # Set the final column of P
        @assert Pcol[i] == 0.0 "Diagonal probability P[$i,$i]=$(Pcol[i]) not zero"
        @inbounds P[:, i] .= Pcol
        beta[i] = betai
    end
    @printf("Mean value of sigma: %.4f\n", mean(sqrt.(1 ./ beta)))
    @printf("Minimum value of sigma: %.4f\n", minimum(sqrt.(1 ./ beta)))
    @printf("Maximum value of sigma: %.4f\n", maximum(sqrt.(1 ./ beta)))
    return P
end

end  # module TSNE
