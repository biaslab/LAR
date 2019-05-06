export ruleVariationalAROutVPPP,
       ruleVariationalARIn1PVPP,
       ruleVariationalARIn2PPVP,
       ruleVariationalARIn3PPPV,
       uvector,
       shift

import LinearAlgebra: Symmetric, tr

order, c, S = Nothing, Nothing, Nothing

diagAR(dim) = Matrix{Float64}(I, dim, dim)

function wMatrix(w, order)
    mW = huge*diagAR(order)
    mW[1, 1] = w
    return mW
end

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
    W = Symmetric(wMatrixunsafeMean(marg_w), order))
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=W*m, w=W)
end

function ruleVariationalARIn1PVPP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: Nothing,
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_w :: ProbabilityDistribution{Univariate})
    ma = unsafeMean(marg_a)
    order == Nothing ? defineOrder(length(ma)) : order
    mA = S+c*ma'
    mw = unsafeMean(marg_w)
    W = Symmetric(unsafeCov(marg_a)*mw+mA'*wMatrixmw, order)*mA)
    xi = mA'*wMatrixmw, order)*unsafeMean(marg_y)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn2PPVP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: Nothing,
                                  marg_w :: ProbabilityDistribution{Univariate})
    my = unsafeMean(marg_y)
    order == Nothing ? defineOrder(length(my)) : order
    mx = unsafeMean(marg_x)
    mw = unsafeMean(marg_w)
    W = Symmetric(unsafeCov(marg_x)*mw+mx*mw*mx')
    xi = (mx*c'*wMatrixmw, order)*my)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn3PPPV(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_w :: Nothing)

    ma = unsafeMean(marg_a)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    B = unsafeCov(marg_y)[1, 1] + my[1]*my'[1] - 2*my[1]*ma'*mx + ma'*(unsafeCov(marg_x)+mx*mx')*ma + mx'*unsafeCov(marg_a)*mx
    Message(Gamma, a=3/2, b= B/2)
end
