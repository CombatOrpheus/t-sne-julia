module TSNE

using Printf: @printf
using LinearAlgebra
using Statistics: mean, mean!

function Hbeta!(P::AbstractVector{T}, D::AbstractVector{T},
     beta=1.0) where T<:Number
    @inbounds P .= exp.(-D .* beta)
    sumP = sum(P)
    H = log(sumP) + beta * dot(D, P) / sumP
    @inbounds P ./= sumP
    return H
end

function diagzero!(X::AbstractMatrix{T}) where T<:Number
    n = zero(T)
    for i in size(X, 1)
        X[i, i] = n
    end
end

function pca(X::AbstractMatrix, ndims::Integer = 50)
    (n, d) = size(X)
    (d <= ndims) && return X
    Y = X .- mean(X, dims=1)
    C = Symmetric((Y' * Y) ./ (n-1))
    Ceig = eigen(C, (d-ndims+1):d) # take eigvects for top ndims largest eigvals
    return Y * reverse(Ceig.vectors, dims=2)
end

function d2p(D::AbstractMatrix{T}, perplexity=15, tol=14e-4) where T<:Number
    # Initializing variables
    n = size(D, 1)          #   Number of instances
    P = zeros(n, n)         #   Empty probability Matrix
    beta = fill(one(T), n)  #   Empty precision vector
    logU = log(perplexity)  #   Log of perplexity
    Pcol = fill(zero(T), n)
    Di = fill(zero(T), n)

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
        copyto!(Di, view(D, :, i))
        Di[i] = prevfloat(Inf) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0
        minD = minimum(Di) # distance of i-th point to its closest neighbour
        @inbounds Di .-= minD # entropy is invariant to offsetting Di, which helps to avoid overflow
        H = Hbeta!(Pcol, Di, betai)

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
            H = Hbeta!(Pcol, Di, betai)
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

function tsne_X(X::AbstractMatrix{T}, no_dims::Integer=2,
              initial_dims::Integer=50, perplexity::Number = 30.0) where T<:Number
    # Normalize input data
    X .-= minimum(X)
    X ./= maximum(X)
    X .-= mean(X, dims=1)

    # Preprocess using PCA
    X = pca(X, initial_dims)

    # Compute pairwise distance matrix
    sum_X = sum(X .^ 2, dims=2)
    D = -2.0 .* (X * X')
    D .+= sum_X .+ sum_X'

    # Compute joint probabilities
    P = d2p(D, perplexity, 1e-5)

    # Run t-SNE
    Y = tsne(P, no_dims)
end

function tsne(P::AbstractMatrix{T}, no_dims::Integer) where T<:Number
    # Initialize some variables
    n = size(P, 1)                                     # Number of instances
    momentum = 0.5                                     # Initial momentum
    final_momentum = 0.8                               # Value to which momentum is changed
    mom_switch_iter = 250                              # Iteration at which momentum is changed
    stop_lying_iter = 100                              # Iteration at which lying about P-values is stopped
    max_iter = 1000                                    # Maximum number of iterations
    epsilon = 500                                      # Initial learning rate
    min_gain = .01                                     # Minimum gain for delta-bar-delta
    realmin = prevfloat(-Inf)

    # Make sure P-vals are set properly
    diagzero!(P)                                       # Set diagonal to zero
    P .= 0.5 .* (P .+ P')                              # Symmetrize P-values
    P .= max.(P ./ sum(P), realmin)                    # Make sure P-values sum to one
    #TODO: Find a way to calculate the KL Divergence error
    # kl_const = sum(P .* log.(P))                       # Constant in KL divergence
    P .= 4.0 .* P                                        # Lie about the P-vals to find better local minima

    # Pre-allocating matrixes
    Y = .0001 .* randn(n, no_dims)
    iY = fill(zero(T), n, no_dims)                      #  Momentum vector
    dY = fill(zero(T), n, no_dims)                      #  Gradient vector
    gains = fill(one(T), n, no_dims)
    sum_Y = fill(one(T), n, 1)
    sum_L = fill(one(T), 1, n)
    num = fill(one(T), n, n)
    Y_mean = fill(one(T), 1, no_dims)
    Y_mul = fill(zero(T), n, n)
    Q = fill(zero(T), n, n)
    L = fill(zero(T), n, n)

    for i in 1:max_iter
        # Compute joint probability that point i and j are neighbors
        sum!(sum_Y, Y .^ 2)
        @inbounds num .= 1 ./(1.0 .+ (sum_Y .+ (sum_Y' .+ (-2.0 .* (Y * Y')))))
        diagzero!(num)                                     # Set diagonal to zero
        Q = max.(num./sum(num), realmin)                  # Normalize to get probabilities

        # Compute the gradients
        @inbounds L .= (P .- Q) .* num
        sum!(sum_L, L)
        @inbounds dY .= -4.0 .* (Diagonal(sum_L) .- L) * Y
        # BLAS.symm!('L', 'U', -4.0, L, Y, 0.0, dY)

        # Update the solution
        gains .= (gains .+ .2) .* (sign.(dY) .!= sign.(iY)) .+         # note that the y_grads are actually -y_grads
                 (gains .* .8) .* (sign.(dY) .== sign.(iY))
        gains .= max.(gains, min_gain)
        iY .= momentum * iY - epsilon * (gains .* dY)
        Y .= Y .+ iY
        Y .= Y .- mean(Y, dims=1)

        # Update the momentum if necessary
        if i == mom_switch_iter
            momentum = final_momentum
        end
        if i == stop_lying_iter
            P .= 4.0 .* P
        end
        if i % 100 == 0
            # cost = kl_const - sum(P .* log.(Q))
            # @printf("Iteration %d: error is %.4f\n", i, cost)
            @printf("Iteration %d of %d\n", i, max_iter)
        end
    end

    return Y
end
end  # module TSNE
