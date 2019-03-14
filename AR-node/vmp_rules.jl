export ruleVariationalAROutVPPP,
       ruleVariationalARIn1PVPP,
       ruleVariationalARIn2PPVP,
       ruleVariationalARIn3PPPV,
       uvector,
       shift

import LinearAlgebra: Symmetric, tr

order, c, S = Nothing, Nothing, Nothing

diagAR(dim) = Matrix{Float64}(I, dim, dim)

function shift(dim)
    S = diagAR(dim)
    for i in dim:-1:2
           S[i,:] = S[i-1, :]
    end
    S[1, :] = zeros(dim)
    return S
end

function uvector(dim, pos=1)
    u = zeros(dim)
    u[pos] = 1
    return u
end

function defineOrder(dim)
    global order, c, S
    order = dim
    c = uvector(order)
    S = shift(order)
end

function ruleVariationalAROutVPPP(marg_y :: Nothing,
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_w :: ProbabilityDistribution{Univariate})

    ma = unsafeMean(marg_a)
    order == Nothing ? defineOrder(length(ma)) : order
    mA = S+c*ma'
    m = mA*unsafeMean(marg_x)
    W = Symmetric(unsafeMean(marg_w)*diagAR(order))
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=W*m, w=W)
end

function ruleVariationalARIn1PVPP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: Nothing,
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_w :: ProbabilityDistribution{Univariate})
    ma = unsafeMean(marg_a)
    order == Nothing ? defineOrder(length(ma)) : order
    mA = S+c*ma'
    D = unsafeCov(marg_a)+mA'*mA
    mw = unsafeMean(marg_w)
    xi = mw*mA'*unsafeMean(marg_y)
    W = Symmetric(mw*D)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn2PPVP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: Nothing,
                                  marg_w :: ProbabilityDistribution{Univariate})
    my = unsafeMean(marg_y)
    order == Nothing ? defineOrder(length(my)) : order
    mx = unsafeMean(marg_x)
    D = unsafeCov(marg_x)+mx*mx'
    mw = unsafeMean(marg_w)
    xi = mw*(mx*c'*my)
    W = Symmetric(mw*D)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn3PPPV(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_w :: Nothing)

    mA = S+c*unsafeMean(marg_a)'
    ma = unsafeMean(marg_a)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    B = tr(unsafeCov(marg_y) + my*my' - 2*my*mx'*mA' + mx*mx'*unsafeCov(marg_a) + (S'*S+ma*ma')*(unsafeCov(marg_x)+mx*mx'))
    Message(Gamma, a=3/2, b= B/2)
end
