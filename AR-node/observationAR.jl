import ForneyLab: @composite, DotProduct, GaussianMeanPrecision
export observationAR, ruleVBobservationARIn2

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