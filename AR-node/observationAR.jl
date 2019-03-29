import ForneyLab: @composite, DotProduct, GaussianMeanPrecision, unsafeLogMean, unsafeMean, unsafeCov
export observationAR, ruleVBobservationARIn2, freeEnergy

@composite observationAR (y, x, z) begin
    c = zeros(ARorder); c[1] = 1.0
    @RV w = dot(x, c)
    @RV y ~ GaussianMeanPrecision(w, z)
end

@naiveVariationalRule(:node_type     => observationAR,
                      :outbound_type => Message{GaussianMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VBobservationARIn2)


function ruleVBobservationARIn2(msg_out::ProbabilityDistribution{V},
                                msg_m::Nothing,
                                msg_w::ProbabilityDistribution{V}) where V<:Univariate
    c = zeros(ARorder); c[1] = 1.0
    d = length(c)
    xi = c * msg_w.params[:m]*msg_out.params[:m]
    w = c * msg_w.params[:m] * c'
    w += tiny*diageye(size(w)[1]) # Ensure w is invertible

    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=w)
end


# Average energy functional
function averageEnergy(::Type{observationAR}, marg_out::ProbabilityDistribution{Univariate, PointMass}, marg_mean::ProbabilityDistribution{Multivariate}, marg_prec::ProbabilityDistribution{Univariate, PointMass})
    mx = unsafeMean(marg_mean)
    Vx = unsafeCov(marg_mean)
    order = length(mx)
    - order/2*log(2*pi) + 0.5*unsafeLogMean(marg_prec)
    + 0.5*unsafeMean(marg_prec)*(unsafeMean(marg_out) - 2*mx[1] - (Vx + mx*mx')[1])
end
