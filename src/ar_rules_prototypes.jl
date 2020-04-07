@naiveVariationalRule(:node_type     => Autoregressive,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalAROutNPPP)


@naiveVariationalRule(:node_type     => Autoregressive,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARIn1PNPP)

@naiveVariationalRule(:node_type     => Autoregressive,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalARIn2PPNP)

@naiveVariationalRule(:node_type     => Autoregressive,
                      :outbound_type => Message{Gamma},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalARIn3PPPN)

# Structured updates

@structuredVariationalRule(:node_type     => Autoregressive,
                           :outbound_type => Message{GaussianMeanVariance},
                           :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution),
                           :name          => SVariationalAROutNPPP)

@structuredVariationalRule(:node_type     => Autoregressive,
                           :outbound_type => Message{GaussianMeanVariance},
                           :inbound_types => (Message{Gaussian}, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                           :name          => SVariationalARIn1PNPP)

@structuredVariationalRule(:node_type     => Autoregressive,
                           :outbound_type => Message{GaussianMeanVariance},
                           :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution),
                           :name          => SVariationalARIn2PPNP)

@structuredVariationalRule(:node_type     => Autoregressive,
                           :outbound_type => Message{Gamma},
                           :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing),
                           :name          => SVariationalARIn3PPPN)

@marginalRule(:node_type => Autoregressive,
              :inbound_types => (Message{Gaussian}, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution),
              :name => MGaussianMeanVarianceGGGD)
