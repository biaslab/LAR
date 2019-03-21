import ForneyLab: @composite, DotProduct, GaussianMeanPrecision, unsafeMeanCov
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
    0
    # (m_mean, v_mean) = unsafeMeanCov(marg_mean)
    # m_out = marg_out.params[:m]
    #
    # 0.5*log(2*pi) -
    # 0.5*log(marg_out.params[:m]) +
    # 0.5*marg_out.params[:m]*(v_mean[1] + (m_out - m_mean[1])^2)
end
