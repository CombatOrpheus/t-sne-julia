module TSNE

using LinearAlgebra, ProgressBars
using Statistics: mean, mean!
using Printf: @printf, @sprintf

export tsne

"""
    Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)

Compute the point perplexities `P` given its squared distances to the other points `D`
ans the precsion of Gaussian distribution `beta`.
"""
function Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)
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
function x2p(X::AbstractMatrix{T}, tol::Number = 1e-5,
     perplexity::Number = 30.0; max_iter::Integer = 50,
	 progress = true) where T<: Number
    # Initializing some variables
    n, d = size(X)
    sum_X = sum(x -> x^2, X, dims=2)
	D = randn(n, n)
	BLAS.gemm!('N', 'T', -2.0, X, X, 0.0, D)
    D .+= sum_X .+ sum_X'
    P = fill(zero(T), n, n)
    beta = fill(one(T), n)
    logU = log(perplexity)
    Di = fill(zero(T), n)
    thisP = similar(Di)

    # Loop over all datapoints
	if progress
		pb = ProgressBar(1:n)
		set_description(pb, "Computing P-values...")
	else
		pb = 1:n
	end
    for i in pb

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

        while abs(Hdiff) > tol && tries < max_iter
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
    @printf("Mean σ: %f\n", mean(sqrt.(1.0 ./ beta)))
    return P
end

"""
    pca(X::AbstractMatrix, ndims::Integer = 50)

Run PCA on `X` to reduce the number of its dimensions to `ndims`.
"""
function pca(X::AbstractMatrix, ndims::Integer = 50)
    (n, d) = size(X)
    (d <= ndims) && return X
    Y = X .- mean(X, dims=1)
    C = Symmetric((Y' * Y) ./ (n-1))
    Ceig = eigen(C, (d-ndims+1):d) # take eigvects for top ndims largest eigvals
    return Y * reverse(Ceig.vectors, dims=2)
end

"""
    tsne(X::AbstractMatrix{T}, no_dims=2, initial_dims=50,
     max_iter::Integer=1000, perplexity=30.0) where T<:Number

Apply t-SNE (t-Distributed Stochastic Neighbor Embedding) to `X`,
i.e. embed its points into `ndims` dimensions preserving cllose neighbours.

Returns the `point×ndims` matrix of calculated embedded coordinates.

Different from original implementation: the default is not to use PCA for initialization.

### Arguments
	* `no_dims` determines the number og dimensions in the final t-SNE embedding
	* `initial_dims` the number of dimensions of the dataset after apllying
	PCA to initialize the solution
	* `max_iter` how many iterations of t-SNE to perform
	* `perplexity` the number of "effective neighbours" of a datapoint,
	usually increases with the amount of points in the dataset. Typical
	values are between 5 and 50, the default is 30
	* `min_gain`, `eta`, `cheat_scale`, `initial_momentum`, `final_momentum`,
	`stop_cheat_iter`, `momentum_switch_iter` low level parameters of t-SNE optimization
"""
function tsne(X::AbstractMatrix{T}, no_dims=2, initial_dims::Integer = 50,
     max_iter::Integer = 1000, perplexity::Number = 30.0;
	 initial_momentum::Number = 0.5, final_momentum = 0.8, eta::Integer = 500,
	 min_gain::Number = 0.01, cheat_scale::Number = 4.0, progress = true,
	 stop_cheat_iter::Integer = 100, momentum_switch_iter::Integer = 20) where T<:Number

	X = pca(X, initial_dims)
    n, d = size(X)
    Y = randn(n, no_dims)
    dY = fill(zero(T), n, no_dims)    # gradient vector
    iY = fill(zero(T), n, no_dims)    # momentum vector
    gains = fill(one(T), n, no_dims)  # how much momentum is affected by gradient
    P = x2p(X, 1e-5, perplexity)
    P .+= P'						  # symmetrization
    P .*= cheat_scale/sum(P)		  # early exaggeration + normalization
    P .= max.(P, 1e-12)

    # Pre-allocating some matrixes
	L = similar(P)					  # temp matrix for Student-t and gradient steps
    Q = similar(P)					  # temp matrix with low dimensional probabilities
	sum_Y = fill(zero(T), n)
	Y_mean = fill(zero(T), 1, no_dims)
	error = similar(P)
	last_error = NaN

    # Run iterations
	pb = progress ? ProgressBar(1:max_iter) : 1:max_iter
    for iter in pb
        # Compute pairwise affinities
        sum!(x -> x^2, sum_Y, Y)
		# L = 2YY'
		BLAS.gemm!('N', 'T', -2.0, Y, Y, 0.0, L)
		# Student-t Distribution
		L .= 1.0 ./(1.0 .+ sum_Y .+ sum_Y' .+ L)
		fill!(sum_Y, 0.0)
        inv_sum_Q = 1.0/sum(L)
		@inbounds for j = 1:size(Q, 2)
			Pj = view(P, :, j)
			Qj = view(Q, :, j)
			Lj = view(L, :, j)
			# Diagonal should be zero
			Lj[j] = 0.0
			@inbounds for i = 1:size(Lj, 1)
				Qj[i] = max(Lj[i] * inv_sum_Q, 1e-12)
				# Reusing L for gradient step
				@fastmath Lj[i] *= (Pj[i] - Qj[i])
			end
			# Reusing sum_Y for column sums
			sum_Y .+= Lj
		end
		# Compute gradient
		@inbounds for (i , ldiag) in enumerate(sum_Y)
			L[i, i] -= ldiag
		end
		# dY = -4LY
		BLAS.gemm!('N', 'N', -4.0, L, Y, 0.0, dY)
        # Perform the update
        momentum = ifelse(iter < momentum_switch_iter, initial_momentum, final_momentum)
        @inbounds for i in eachindex(gains)
            gains[i] = max(ifelse(((dY[i] > 0.) == (iY[i] > 0.)),
                                gains[i] * 0.8,
                                gains[i] + 0.2),
                                min_gain)
            iY[i] = momentum * iY[i] - eta * (gains[i] * dY[i])
            Y[i] += iY[i]
        end
        @inbounds Y .-= mean!(Y_mean, Y)

        # Compute current value of cost function
        if iter % 50 == 0
            @. error = P * log(P / Q)
			last_error = sum(error)
        end

		progress && set_description(pb, string(@sprintf("Error: %.4f", last_error)))

        # Stop lying about P-values
        if iter == stop_cheat_iter
            P ./= 4.0
        end
    end
    # Return solution
    return Y
end

end  # module TSNE
