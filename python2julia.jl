module TSNE

using LinearAlgebra
using Statistics: mean, mean!
using Printf: @printf

export tsne
"""
    Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)

Compute the point perplexities `P` given its squared distances to the other points `D`
ans the precsion of Gaussian distribution `beta`.
"""
@inline function Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)
    @inbounds P .= exp.(-D .* beta)
    sumP = sum(P)
    H = log(sumP) + beta * dot(D, P) / sumP
    @inbounds P ./= sumP
    return H
end

"""
    x2p(D::AbstractMatrix{T}, tol::Number = 1e-5, perplexity::Number = 30.0)

Convert `n×n` squared distances matrix `D` into `n×n` perplexities matrix `P`.
Performs a binary seach to get P-values in such a way that each conditional
Gaussian has the same perplexity.
"""
function x2p(X::AbstractMatrix{T}, tol::Number =1e-5,
     perplexity::Number =30.0) where T<: Number
    # Initializing some variables
    n, d = size(X)
    sum_X = sum(X .^ 2, dims=2)
    D = sum_X .+ (sum_X .+ (-2.0 .* (X * X')))'
    P = fill(zero(T), n, n)
    beta = fill(one(T), n)
    logU = log(perplexity)
    Di = fill(zero(T), n)
    thisP = similar(Di)

    # Loop over all datapointa
    for i in 1:n

        if i % 500 == 0
            @printf("Computing P-values for point %d of %d...\n", i, n)
        end

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = 0.0
        betamax = Inf
        betai = 1.0

        copyto!(thisP, view(P, i, :))
        copyto!(Di, view(D, :, i))
        Di[i] = prevfloat(Inf) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0
        minD = minimum(Di) # distance of i-th point to its closest neighbour
        @inbounds Di .-= minD # entropy is invariant to offsetting Di, which helps to avoid overflow
        H = Hbeta!(thisP, Di, betai)

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
            H = Hbeta!(thisP, Di, betai)
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

"""
    pca(X::AbstractMatrix, ndims::Integer = 50)

Run PCA on `X` to reduce the number of its dimensions to `ndims`.
"""
@inline function pca(X::AbstractMatrix, ndims::Integer = 50)
    (n, d) = size(X)
    (d <= ndims) && return X
    Y = X .- mean(X, dims=1)
    C = Symmetric((Y' * Y) ./ (n-1))
    Ceig = eigen(C, (d-ndims+1):d) # take eigvects for top ndims largest eigvals
    return Y * reverse(Ceig.vectors, dims=2)
end

#TODO: Consider removing this functions since its very simple and could be
#manualy written whre needed.
function inplace_max!(A::AbstractVecOrMat, comp::Number)
    @simd for i in eachindex(A)
        @inbounds A[i] = ifelse(A[i] > comp,
                                A[i],
                                comp)
    end
end

"""
    tsne(X::AbstractMatrix{T}, no_dims=2, initial_dims=50,
     perplexity=30.0) where T<:Number

Apply t-SNE (t-Distributed Stochastic Neighbor Embedding) to `X`,
i.e. embed its points into `ndims` dimensions preserving cllose neighbours.

Returns the `point×ndims` matrix of calculated embedded coordinates.

Different from original implementation: the default is not to use PCA for initialization.
"""
#TODO: Add the tsne variables to the function signature and make them be
#keyword arguments.
function tsne(X::AbstractMatrix{T}, no_dims=2, initial_dims::Integer = 50,
     perplexity::Number = 30.0; max_iter::Integer = 1000,
     initial_momentum::Number = 0.5, final_momentum = 0.8, eta::Integer = 500,
     min_gain::Number = 0.01, cheat_scale::Number = 4.0) where T<:Number

    X = pca(X, initial_dims)
    n, d = size(X)
    Y = randn(n, no_dims)
    dY = fill(zero(T), n, no_dims)              # gradient vector
    iY = fill(zero(T), n, no_dims)              # momentum vector
    gains = fill(one(T), n, no_dims)            # how much momentum is affected bt gradient
    sum_Y = fill(zero(T), n)
    sum_YY = fill(one(T), n , n)

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P .+= P'
    P .*= cheat_scale/sum(P)					# early exaggeration + normalization
    inplace_max!(P, 1e-12)
    L = similar(P)
    Y_mean = fill(zero(T), 1, no_dims)

    # Pre-allocating some matrixes
    num = fill(one(T), n, n)
    inter_num = similar(num)
    Q = fill!(similar(num), zero(T))
    gradient = fill(one(T), n, no_dims)
    C = [0.0]
    error = fill!(similar(P), one(T))
    Q_part = [0.0]

    # Run iterations
    for iter in 1:max_iter

        # Compute pairwise affinities
        sum!(x -> x^2, sum_Y, Y)
        # inter_num = -2YY'
        BLAS.gemm!('N', 'T', -2.0, Y, Y, 0.0, inter_num)
        transpose!(num, inter_num .+= sum_Y)
        @. num = 1.0 /(1.0 + (num + sum_Y))
        for i in 1:n
            num[i,i] = 0.0
        end
        Q .= num ./ sum!(Q_part, num)
        inplace_max!(Q, 1e-12)

        # Compute gradient
        L .= (P .- Q) .* num
        for i in 1:n
            Li = view(L, :, i)
            dYi = view(dY, i, :)'
            Yi = view(Y, i, :)'
            gradient .= Li .* (Yi .- Y)
            sum!(dYi, gradient)
        end
        # Perform the update
        momentum = ifelse(iter < 20, initial_momentum, final_momentum)
        for i in eachindex(gains)
            gains[i] = max(ifelse(((dY[i] > 0.) == (iY[i] > 0.)),
                                gains[i] * 0.8,
                                gains[i] + 0.2),
                                min_gain)
            iY[i] = momentum * iY[i] - eta * (gains[i] * dY[i])
            Y[i] += iY[i]
        end
        @inbounds Y .-= mean!(Y_mean, Y)

        # Compute current value of cost function
        if (iter + 1) % 100 == 0
            # C = sum(P .* log.(P ./ Q))
            @inbounds @. error = P * log(P / Q)
            sum!(C, error)
            @printf("Iteration %d: error is %f\n", iter + 1, C[1])
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
