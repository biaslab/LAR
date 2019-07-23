import LinearAlgebra: I, Hermitian, tr
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision

export ruleVariationalAROutNPPP,
       ruleVariationalARIn1PNPP,
       ruleVariationalARIn2PPNP,
       ruleVariationalARIn3PPPN,
       uvector,
       shift,
       wMatrix

order, c, S = Nothing, Nothing, Nothing

diagAR(dim) = Matrix{Float64}(I, dim, dim)

function wMatrix(γ, order)
    mW = huge*diagAR(order)
    mW[1, 1] = γ
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

function ruleVariationalAROutNPPP(marg_y :: Nothing,
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_θ :: ProbabilityDistribution{Multivariate},
                                  marg_γ :: ProbabilityDistribution{Univariate})
    mθ = unsafeMean(marg_θ)
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order
    mA = S+c*mθ'
    m = mA*unsafeMean(marg_x)
    W = Hermitian(wMatrix(unsafeMean(marg_γ), order))
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=W*m, w=W)
end

function ruleVariationalARIn1PNPP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: Nothing,
                                  marg_θ :: ProbabilityDistribution{Multivariate},
                                  marg_γ :: ProbabilityDistribution{Univariate})
    mθ = unsafeMean(marg_θ)
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order
    mA = S+c*mθ'
    mγ = unsafeMean(marg_γ)
    W = Hermitian(unsafeCov(marg_θ)*mγ+mA'*wMatrix(mγ, order)*mA)
    xi = mA'*wMatrix(mγ, order)*unsafeMean(marg_y)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn2PPNP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_θ :: Nothing,
                                  marg_γ :: ProbabilityDistribution{Univariate})
    my = unsafeMean(marg_y)
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order
    mx = unsafeMean(marg_x)
    mγ = unsafeMean(marg_γ)
    W = Hermitian(unsafeCov(marg_x)*mγ+mx*mγ*mx')
    xi = (mx*c'*wMatrix(mγ, order)*my)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn3PPPN(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_θ :: ProbabilityDistribution{Multivariate},
                                  marg_γ :: Nothing)
    mθ = unsafeMean(marg_θ)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    B = unsafeCov(marg_y)[1, 1] + my[1]*my[1] - 2*my[1]*mθ'*mx + mθ'*(unsafeCov(marg_x)+mx*mx')*mθ + mx'*unsafeCov(marg_θ)*mx
    Message(Gamma, a=3/2, b=B/2)
end
