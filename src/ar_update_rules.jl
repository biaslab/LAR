import LinearAlgebra: I, Hermitian, tr
import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, VariateType

export ruleVariationalAROutNPPP,
       ruleVariationalARIn1PNPP,
       ruleVariationalARIn2PPNP,
       ruleVariationalARIn3PPPN,
       ruleSVariationalAROutNPPP,
       ruleSVariationalARIn1PNPP,
       ruleSVariationalARIn2PPNP,
       ruleSVariationalARIn3PPPN,
       ruleMGaussianMeanVarianceGGGD,
       ruleSVBGaussianMeanPrecisionMGVD,
       uvector,
       shift,
       wMatrix

function wMatrix(γ, order)
    mW = huge*Matrix{Float64}(I, order, order)
    mW[1, 1] = γ
    return mW
end

function transition(γ, order)
    V = zeros(order, order)
    V[1] = 1/γ
    return V
end

function shift(dim)
    S = Matrix{Float64}(I, dim, dim)
    for i in dim:-1:2
           S[i,:] = S[i-1, :]
    end
    S[1, :] = zeros(dim)
    return S
end

function uvector(dim, pos=1)
    u = zeros(dim)
    u[pos] = 1
    return dim == 1 ? u[pos] : u
end

function defineCS(order)
    uvector(order), shift(order)
end

# Meanfield updates
function ruleVariationalAROutNPPP(marg_y :: Nothing,
                                  marg_x :: ProbabilityDistribution{V},
                                  marg_θ :: ProbabilityDistribution{V},
                                  marg_γ :: ProbabilityDistribution{Univariate}) where V<:VariateType
    mθ = unsafeMean(marg_θ)
    c, S = defineCS(length(mθ))
    # The user can specify AR(1) as Multivariate distribution
    if V == Multivariate
        mθ = S+c*mθ'
        mW = wMatrix(unsafeMean(marg_γ), length(c))
    else
        mW = unsafeMean(marg_γ)
    end
    m = mθ*unsafeMean(marg_x)
    Message(V, GaussianWeightedMeanPrecision, xi=mW*m, w=mW)
end

function ruleVariationalARIn1PNPP(marg_y :: ProbabilityDistribution{V},
                                  marg_x :: Nothing,
                                  marg_θ :: ProbabilityDistribution{V},
                                  marg_γ :: ProbabilityDistribution{Univariate}) where V<:VariateType
    mθ, Vθ = unsafeMeanCov(marg_θ)
    my = unsafeMean(marg_y)
    c, S = defineCS(length(mθ))
    # The user can specify AR(1) as Multivariate distribution
    if V == Multivariate
        mθ = S+c*mθ'
        mW = wMatrix(unsafeMean(marg_γ), length(c))
    else
        mW = unsafeMean(marg_γ)
    end
    W = mθ'*mW*mθ + Vθ*unsafeMean(marg_γ)
    xi = mθ'*mW*my
    Message(V, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn2PPNP(marg_y :: ProbabilityDistribution{V},
                                  marg_x :: ProbabilityDistribution{V},
                                  marg_θ :: Nothing,
                                  marg_γ :: ProbabilityDistribution{Univariate}) where V<:VariateType
    my = unsafeMean(marg_y)
    c, S = defineCS(length(my))
    mW = V == Multivariate ? wMatrix(unsafeMean(marg_γ), length(c)) : unsafeMean(marg_γ)

    mx = unsafeMean(marg_x)
    W = unsafeMean(marg_γ)*(unsafeCov(marg_x)+mx*mx')
    xi = mx*c'*mW*my
    Message(V, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn3PPPN(marg_y :: ProbabilityDistribution{V},
                                  marg_x :: ProbabilityDistribution{V},
                                  marg_θ :: ProbabilityDistribution{V},
                                  marg_γ :: Nothing) where V<:VariateType
    mθ, Vθ = unsafeMeanCov(marg_θ)
    my, Vy = unsafeMeanCov(marg_y)
    mx, Vx = unsafeMeanCov(marg_x)
    B = Vy[1, 1] + my[1]*my[1] - 2*my[1]*mθ'*mx + mx'*Vθ*mx + mθ'*(Vx+mx*mx')*mθ
    Message(Univariate, Gamma, a=3/2, b=B/2)
end


# Structured updates
function ruleSVariationalAROutNPPP(marg_y :: Nothing,
                                   msg_x ::  Message{F, V},
                                   marg_θ :: ProbabilityDistribution{V},
                                   marg_γ :: ProbabilityDistribution{Univariate}) where {F<:Gaussian, V<:VariateType}
    mθ, Vθ = unsafeMeanCov(marg_θ)
    mx, Vx = unsafeMeanCov(msg_x.dist)
    c, S = defineCS(length(mθ))
    # The user can specify AR(1) as Multivariate distribution
    if V == Multivariate
       mθ = S+c*mθ'
       mV = transition(unsafeMean(marg_γ), length(c))
    else
       mV = inv(unsafeMean(marg_γ))
    end
    D = inv(Vx) + unsafeMean(marg_γ)*Vθ
    my = mθ*inv(D)*inv(Vx)*mx
    Vy = mθ*inv(D)*mθ' + mV
    Message(V, GaussianMeanVariance, m=my, v=Vy)
end

function ruleSVariationalARIn1PNPP(msg_y :: Message{F, V},
                                   marg_x :: Nothing,
                                   marg_θ :: ProbabilityDistribution{V},
                                   marg_γ :: ProbabilityDistribution{Univariate}) where {F<:Gaussian, V<:VariateType}
    mθ, Vθ = unsafeMeanCov(marg_θ)
    my, Vy = unsafeMeanCov(msg_y.dist)
    c, S = defineCS(length(mθ))
    # The user can specify AR(1) as Multivariate distribution
    if V == Multivariate
        mθ = S+c*mθ'
        mV = transition(unsafeMean(marg_γ), length(c))
    else
        mV = inv(unsafeMean(marg_γ))
    end
    D = mθ'*inv(Vy + mV)*mθ + unsafeMean(marg_γ)*Vθ
    Vx = inv(D)
    mx = inv(D)*mθ'*inv(Vy + mV)*my
    Message(V, GaussianMeanVariance, m=mx, v=Vx)
end

function ruleSVariationalARIn2PPNP(marg_xy :: ProbabilityDistribution{Multivariate},
                                   marg_θ  :: Nothing,
                                   marg_γ  :: ProbabilityDistribution{Univariate})
    c, S = defineCS(div(length(marg_xy.params[:m]), 2))
    order = length(c)

    marg_xy = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, marg_xy)

    my, Vy = marg_xy.params[:m][1:order], marg_xy.params[:v][1:order,1:order]
    mx, Vx = marg_xy.params[:m][order+1:end], marg_xy.params[:v][order+1:end, order+1:end]
    Vxy = marg_xy.params[:v][order+1:end,1:order]
    mγ = unsafeMean(marg_γ)

    D = mγ*(Vx + mx*mx')
    mθ, Vθ = inv(D)*(Vxy + mx*my')*wMatrix(mγ, order)*c, inv(D)

    # Check if parameters are of Array type
    V = Multivariate
    if order == 1
        mθ, Vθ = mθ[1], Vθ[1]
        V = Univariate
    end

    Message(V, GaussianMeanVariance, m=mθ, v=Vθ)
end

function ruleSVariationalARIn3PPPN(marg_xy :: ProbabilityDistribution{V1},
                                   marg_θ  :: ProbabilityDistribution{V2},
                                   marg_γ  :: Nothing) where {V1<:VariateType, V2<:VariateType}

    c, S = defineCS(div(length(marg_xy.params[:m]), 2))
    order = length(c)

    mθ, Vθ = V2 == Multivariate ? S+c*unsafeMean(marg_θ)' : unsafeMean(marg_θ), unsafeCov(marg_θ)
    my, Vy = marg_xy.params[:m][1:order], marg_xy.params[:v][1:order,1:order]
    mx, Vx = marg_xy.params[:m][order+1:end], marg_xy.params[:v][order+1:end, order+1:end]
    Vxy = marg_xy.params[:v][order+1:end,1:order]

    B = (Vy + my*my')[1, 1] - 2*(mθ*(Vxy + mx*my'))[1, 1] + (mθ*(Vx + mx*mx')*mθ')[1, 1] + tr(Vθ*(Vx + mx*mx'))
    Message(Univariate, Gamma, a=3/2, b=B/2)
end

function ruleMGaussianMeanVarianceGGGD(msg_y::Message{F1, V},
                                       msg_x::Message{F2, V},
                                       dist_θ::ProbabilityDistribution,
                                       dist_γ::ProbabilityDistribution) where {F1<:Gaussian, F2<:Gaussian, V<:VariateType}

    mθ, Vθ = unsafeMeanCov(dist_θ)
    c, S = defineCS(length(mθ))
    # The user can specify AR(1) as Multivariate distribution
    if V == Multivariate
       mθ = S+c*mθ'
       mW = wMatrix(unsafeMean(dist_γ), length(c))
    else
       mW = unsafeMean(dist_γ)
    end

    b_my = unsafeMean(msg_y.dist)
    b_Vy = unsafeCov(msg_y.dist)
    f_mx = unsafeMean(msg_x.dist)
    f_Vx = unsafeCov(msg_x.dist)

    D = inv(f_Vx) + unsafeMean(dist_γ)*Vθ
    W = [inv(b_Vy)+mW -mW*mθ; -mθ'*mW D+mθ'*mW*mθ]
    m = inv(W)*[inv(b_Vy)*b_my; inv(f_Vx)*f_mx]
    return ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=m, v=inv(W))
end

# NOTE: This function is not safe for AR > 1
# function ruleMGaussianMeanVarianceGGGD(msg_y::Message{F1, V},
#                                        msg_x::Message{F2, V},
#                                        dist_θ::ProbabilityDistribution,
#                                        dist_γ::ProbabilityDistribution) where {F1<:Gaussian, F2<:Gaussian, V<:VariateType}
#
#     mθ, Vθ = unsafeMeanCov(dist_θ)
#     c, S = defineCS(length(mθ))
#     # The user can specify AR(1) as Multivariate distribution
#     mγ = unsafeMean(dist_γ)
#     if V == Multivariate
#        mθ = S+c*mθ'
#        mV = transition(mγ, length(c))
#     else
#        mV = inv(mγ)
#     end
#
#     b_my = unsafeMean(msg_y.dist)
#     b_Vy = unsafeCov(msg_y.dist)
#     f_mx = unsafeMean(msg_x.dist)
#     f_Vx = unsafeCov(msg_x.dist)
#
#     E = mV - mV*inv(b_Vy + mV)*mV
#     F = mV + mV*inv(mθ)'*(inv(f_Vx) + mγ*Vθ)*inv(mθ)'*mV
#     ABDC = E - E*inv(F + E)*E
#     BD = -inv(mθ)' + inv(mθ)'*inv(inv(mθ)*mV*inv(mθ)' + inv((inv(f_Vx) + mγ*Vθ)))*inv(mθ)*mV*inv(mθ)'
#     DC =  -inv(mθ) + inv(mθ)*mV*inv(mθ)'*inv(inv(mθ)*mV*inv(mθ)' + inv((inv(f_Vx) + mγ*Vθ)))*inv(mθ)
#     D = inv(mθ)*mV*inv(mθ)' - inv(mθ)*mV*inv(mθ)'*inv(inv(mθ)*mV*inv(mθ)' + inv((inv(f_Vx) + mγ*Vθ)))*inv(mθ)*mV*inv(mθ)'
#     invW = [ABDC -ABDC*BD; -DC*ABDC D+DC*ABDC*BD]
#     m = invW*[inv(b_Vy)*b_my; inv(f_Vx)*f_mx]
#     return ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=m, v=invW)
# end
