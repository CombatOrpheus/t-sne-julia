module TSNE

function Hbeta!(Pcol::AbstractMatrix{T}, D::AbstractMatrix{T},
     beta::T=1.0) where T<:Number
    @inbounds P .= exp.(-D .* beta)
    sumP = sum(P)
    H = log(sumP) + beta * sum(D .* P) / sumP
    @inbounds P ./= sumP
end
end  # module TSNE
