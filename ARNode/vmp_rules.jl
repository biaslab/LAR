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

function transition(γ, order)
    V = zeros(order, order)
    V[1] = 1/γ
    return V
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

function shiftD(matrix, γ)
    for i in 2:size(matrix)[1]
        matrix[i, i] = matrix[i - 1, i - 1]
    end
    matrix[1] = γ
    return matrix
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
    #W = Hermitian(mθ'*wMatrix(unsafeMean(marg_γ)*mθ, order))
    W = Hermitian(shiftD(unsafeCov(marg_x), mθ'*unsafeMean(marg_γ)*mθ))
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
    W = Hermitian(mγ*(unsafeCov(marg_θ) + mθ*mθ'))#Hermitian(unsafeCov(marg_θ)*mγ+mA'*wMatrix(mγ, order)*mA)
    xi = mθ*unsafeMean(marg_y)[1]#mA'*wMatrix(mγ, order)*unsafeMean(marg_y)
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
    B = unsafeCov(marg_y)[1, 1] + my[1]*my[1] - 2*my[1]*mθ'*mx + mθ'*(unsafeCov(marg_x)+mx*mx')*mθ + tr(unsafeCov(marg_θ)*unsafeCov(marg_x)) + mx'*unsafeCov(marg_θ)*mx
    Message(Gamma, a=3/2, b=B/2)
end

# Structured updated

function ruleSVariationalAROutNPPP(marg_y :: Nothing,
                                   marg_x :: ProbabilityDistribution{Multivariate},
                                   marg_θ :: ProbabilityDistribution{Multivariate},
                                   marg_γ :: ProbabilityDistribution{Univariate})
    mθ = unsafeMean(marg_θ)
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order
    mA = S+c*mθ'
    mx = unsafeMean(marg_x)
    mγ = unsafeMean(marg_γ)
    Vx = unsafeCov(marg_x)
    Vθ = unsafeCov(marg_θ)
    D = inv(Vx) + mγ*Vθ)
    my = mA*inv(D)*inv(Vx)*mx
    trans = transition(mγ, order)
    Vy = mA*inv(D)*mA' + trans
    Message(Multivariate, GaussianMeanVariance, m=my, v=Vy)
end

function ruleSVariationalARIn1PNPP(marg_y :: ProbabilityDistribution{Multivariate},
                                   marg_x :: Nothing,
                                   marg_θ :: ProbabilityDistribution{Multivariate},
                                   marg_γ :: ProbabilityDistribution{Univariate})
    mθ = unsafeMean(marg_θ)
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order
    my = unsafeMean(marg_y)
    mγ = unsafeMean(marg_γ)
    Vy = unsafeCov(marg_y)
    Vθ = unsafeCov(marg_θ)
    trans = transition(mγ, order)
    G = inv(mA) + inv(mA)*trans*inv(mA)'*inv(inv(mγ)*Vθ + inv(mA)*trans*inv(mA)')*inv(mA)
    mx = G*(my-Vy*inv(trans + Vy)*my)
    Vx = G*(Vy - Vy*inv(trans+Vy)*Vy)*G' + Vy - inv(trans + Vy)*Vy
    Message(Multivariate, GaussianMeanVariance, x=mx, v=Vx)
end

function ruleSVariationalARIn2PPNP(marg_y :: ProbabilityDistribution{Multivariate},
                                   marg_x :: ProbabilityDistribution{Multivariate},
                                   marg_θ :: Nothing,
                                   marg_γ :: ProbabilityDistribution{Univariate})
    mγ = unsafeMean(marg_γ)
    m̂y = unsafeMean(marg_y)
    m̂x = unsafeMean(marg_x)
    V̂y = unsafeCov(marg_y)
    V̂x = unsafeCov(marg_x)
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order
    mx = unsafeMean(marg_x)
    mγ = unsafeMean(marg_γ)
    W = Hermitian(unsafeCov(marg_x)*mγ+mx*mγ*mx')
    xi = (mx*c'*wMatrix(mγ, order)*my)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleSVariationalARIn3PPPN(marg_y :: ProbabilityDistribution{Multivariate},
                                   marg_x :: ProbabilityDistribution{Multivariate},
                                   marg_θ :: ProbabilityDistribution{Multivariate},
                                   marg_γ :: Nothing)
    mθ = unsafeMean(marg_θ)
    mA = S+c*mθ'
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    Vy = unsafeCov(marg_y)
    Vx = unsafeCov(marg_x)
    Vθ = unsafeCov(marg_θ)
    B = (mA*(Vx + mx*mx)'*mA')[1, 1] + tr(Vθ*Vx) + mx'*Vθ*mx
    Message(Gamma, a=3/2, b=B)
end
