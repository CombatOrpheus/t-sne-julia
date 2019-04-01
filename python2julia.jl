module TSNE

using LinearAlgebra
using Statistics: mean, mean!
using Printf: @printf

function Hbeta(D::AbstractVector{T}, beta=1.0) where T<: Number
    P = exp.(-D * beta)
    sumP = sum(P)
    H = log(sumP) + beta * dot(D, P) / sumP
    P ./= sumP
    return H, P
end

function x2p(X::AbstractMatrix{T}, tol=1e-5, perplexity=30.0) where T<: Number
    # Initializing some variables
    n, d = size(X)
    sum_X = sum(X .^ 2, dims=2)
    D = sum_X .+ (sum_X .+ (-2.0 .* (X * X')))'
    P = fill(zero(T), n, n)
    beta = fill(one(T), n)
    logU = log(perplexity)
    Di = fill(zero(T), n)

    # Loop over all datapointa
    for i in 1:n

        if i % 500 == 0
            @printf("Computing P-values for point %d of %d...\n", i, n)
        end

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = 0.0
        betamax = Inf
        betai = 1.0

        copyto!(Di, view(D, :, i))
        Di[i] = prevfloat(Inf) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0
        minD = minimum(Di) # distance of i-th point to its closest neighbour
        @inbounds Di .-= minD # entropy is invariant to offsetting Di, which helps to avoid overflow
        H, thisP = Hbeta(Di, betai)

        Hdiff = H - logU
        tries = 0

        while abs(Hdiff) > tol && tries < 50
            # If not, increase or decrease precision
            if Hdiff > 0.0
                betamin = betai
                betai = isfinite(betamax) ? (betai + betamax)/2 : betai*2
            else
                betamax = betai
                betai = (betai + betamin)/2
            end

            # Recompute the values
            H, thisP = Hbeta(Di, betai)
            Hdiff = H - logU
            tries += 1
        end

        # Set the final row of P
        @inbounds P[i, :] .= thisP
        beta[i] = betai
    end
    # Return final P matrix
    @printf("Mean value of sigma: %f\n", mean(sqrt.(1.0 ./ beta)))
    return P
end

function pca(X::AbstractMatrix, ndims::Integer = 50)
    (n, d) = size(X)
    (d <= ndims) && return X
    Y = X .- mean(X, dims=1)
    C = Symmetric((Y' * Y) ./ (n-1))
    Ceig = eigen(C, (d-ndims+1):d) # take eigvects for top ndims largest eigvals
    return Y * reverse(Ceig.vectors, dims=2)
end

function tsne(X::AbstractMatrix{T}, no_dims=2, initial_dims=50,
     perplexity=30.0) where T<:Number

    X = pca(X, initial_dims)
    n, d = size(X)
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = randn(n, no_dims)
    dY = fill(zero(T), n, no_dims)
    iY = fill(zero(T), n, no_dims)
    gains = fill(one(T), n, no_dims)

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P .+= P'
    P ./= sum(P)
    P .*= 4.									# early exaggeration
    P .= max.(P, 1e-12)
    PQ = similar(P)
    Y_mean = fill(zero(T), 1, no_dims)

    # Run iterations
    for iter in 1:max_iter

        # Compute pairwise affinities
        sum_Y = sum(Y .^ 2, dims=2)
        num = -2.0 .* (Y * Y')
        num .= 1 ./(1.0 .+ (sum_Y .+ (sum_Y .+ num)'))
        # num[1:n, 1:n] .= 0.0
        for i in 1:n
            @inbounds num[i,i] = 0
        end
        Q = num / sum(num)
        Q .= max.(Q, 1e-12)

        # Compute gradient
        @. PQ = P - Q
        for i in 1:n
            sum!(view(dY, i, :), repeat(PQ[:, i] .* num[:, i], 1, no_dims)' * (Y[i, :]' .- Y))
            # dY[i, :] .= sum(repeat(PQ[:, i] .* num[:, i], 1, no_dims)' * (Y[i, :]' .- Y), dims=1)
            # dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        end
        # Perform the update
        momentum = ifelse(iter < 20, initial_momentum, final_momentum)
        @. gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) +
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains .= max.(gains, min_gain)
        @. iY = momentum * iY - eta * (gains * dY)
        Y .+= iY
        Y .-= mean!(Y_mean, Y)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0
            C = sum(P .* log.(P ./ Q))
            @printf("Iteration %d: error is %f\n", iter + 1, C)
        end

        # Stop lying about P-values
        if iter == 100
            P ./= 4.0
        end
    end

    # Return solution
    return Y
end

end  # module TSNE
