import LinearAlgebra: I, Hermitian, tr
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType

export ruleVariationalAROutNPPP,
       ruleVariationalARIn1PNPP,
       ruleVariationalARIn2PPNP,
       ruleVariationalARIn3PPPN,
       ruleSVariationalAROutNPPP,
       ruleSVariationalARIn1PNPP,
       ruleSVariationalARIn2PPNP,
       ruleSVariationalARIn3PPPN,
       ruleMGaussianMeanVarianceGGGD,
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

function ruleVariationalAROutNPPP(marg_y :: Nothing,
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_θ :: ProbabilityDistribution{Multivariate},
                                  marg_γ :: ProbabilityDistribution{Univariate})
    mθ = unsafeMean(marg_θ)
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order
    mA = S+c*mθ'
    m = mA*unsafeMean(marg_x)
    W = wMatrix(unsafeMean(marg_γ), order)
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
    mV = transition(mγ, order)
    my = unsafeMean(marg_y)
    D = inv(mA)*mV*inv(mA') - inv(mA)*mV*inv(mA')*inv(inv(mA)*mV*inv(mA') + inv(mγ*unsafeCov(marg_θ)))*inv(mA)*mV*inv(mA')
    invDz = inv(mA)*my - inv(mA)*mV*inv(mA')*inv(inv(mA)*mV*inv(mA') + mγ*unsafeCov(marg_θ))*inv(mA)*my
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=invDz, w=D)
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
    W = unsafeCov(marg_x)*mγ+mx*mγ*mx'
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
    Vθ = unsafeCov(marg_θ)
    Vy = unsafeCov(marg_y)
    Vx = unsafeCov(marg_x)
    B = Vy[1, 1] + my[1]*my[1] - 2*my[1]*mθ'*mx + mx'*Vθ*mx + mθ'*(Vx+mx*mx')*mθ
    Message(Gamma, a=3/2, b=B/2)
end

function ruleVariationalARIn2PPNP(marg_y :: ProbabilityDistribution{V, PointMass},
                                  marg_x :: ProbabilityDistribution{Multivariate, PointMass},
                                  marg_θ :: Nothing,
                                  marg_γ :: ProbabilityDistribution{Univariate}) where V<:VariateType
    my = unsafeMean(marg_y)
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order
    mx = unsafeMean(marg_x)
    mγ = unsafeMean(marg_γ)
    W =  mx*mγ*mx'
    xi = (mx*c'*wMatrix(mγ, order)*my)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn3PPPN(marg_y :: ProbabilityDistribution{V, PointMass},
                                  marg_x :: ProbabilityDistribution{Multivariate, PointMass},
                                  marg_θ :: ProbabilityDistribution{Multivariate},
                                  marg_γ :: Nothing) where V<:VariateType
    mθ = unsafeMean(marg_θ)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    Vθ = unsafeCov(marg_θ)
    B = my[1]*my[1] - 2*my[1]*mθ'*mx + mx'*(Vθ+mθ'mθ)*mx
    Message(Gamma, a=3/2, b=B/2)
end

# No copying behavior (classical AR)

function ruleVariationalARIn2PPNP(marg_y :: ProbabilityDistribution{Univariate, PointMass},
                                  marg_x :: ProbabilityDistribution{Multivariate, PointMass},
                                  marg_θ :: Nothing,
                                  marg_γ :: ProbabilityDistribution{Univariate})
    my = unsafeMean(marg_y)
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order
    mx = unsafeMean(marg_x)
    mγ = unsafeMean(marg_γ)
    W =  mγ*mx*mx'
    xi = (mγ*mx*my)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
end

function ruleVariationalARIn3PPPN(marg_y :: ProbabilityDistribution{Univariate, PointMass},
                                  marg_x :: ProbabilityDistribution{Multivariate, PointMass},
                                  marg_θ :: ProbabilityDistribution{Multivariate},
                                  marg_γ :: Nothing)
    mθ = unsafeMean(marg_θ)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    Vθ = unsafeCov(marg_θ)
    B = my[1]*my[1] - 2*my[1]*mθ'*mx + mx'*Vθ*mx + mθ'*mx*mx'*mθ
    Message(Gamma, a=3/2, b=B/2)
end

# Structured updates

function ruleSVariationalAROutNPPP(marg_y :: Nothing,
                                   msg_x :: Message{F, Multivariate},
                                   marg_θ :: ProbabilityDistribution{Multivariate},
                                   marg_γ :: ProbabilityDistribution{Univariate}) where F<:Gaussian
    mθ = unsafeMean(marg_θ)
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order
    mA = S+c*mθ'
    mx = unsafeMean(msg_x.dist)
    mγ = unsafeMean(marg_γ)
    Vx = unsafeCov(msg_x.dist)
    Vθ = unsafeCov(marg_θ)
    D = inv(Vx) + mγ*Vθ
    my = mA*inv(D)*inv(Vx)*mx
    trans = transition(mγ, order)
    Vy = mA*inv(D)*mA' + trans
    # Check if parameters are of Array type
    Vy = isa(Vy, Array) ? Vy : [Vy]
    my = isa(my, Array) ? my : [my]
    Message(Multivariate, GaussianMeanVariance, m=my, v=Vy)
end

function ruleSVariationalARIn1PNPP(msg_y :: Message{F, Multivariate},
                                   marg_x :: Nothing,
                                   marg_θ :: ProbabilityDistribution{Multivariate},
                                   marg_γ :: ProbabilityDistribution{Univariate}) where F<:Gaussian
    mθ = unsafeMean(marg_θ)
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order
    mA = S+c*mθ'
    my = unsafeMean(msg_y.dist)
    mγ = unsafeMean(marg_γ)
    Vy = unsafeCov(msg_y.dist)
    Vθ = unsafeCov(marg_θ)
    trans = transition(mγ, order)
    D = mA'*inv(Vy + trans)*mA + mγ*Vθ
    Vx = inv(D)
    mx = inv(D)*mA'*inv(Vy + trans)*my
    Vx = isa(Vx, Array) ? Vx : [Vx]
    mx = isa(mx, Array) ? mx : [mx]
    Message(Multivariate, GaussianMeanVariance, m=mx, v=Vx)
end

function ruleSVariationalARIn2PPNP(marg_xy :: ProbabilityDistribution{Multivariate},
                                   marg_θ :: Nothing,
                                   marg_γ :: ProbabilityDistribution{Univariate})
    order == Nothing ? defineOrder(div(length(marg_xy.params[:m]), 2)) : order != div(length(marg_xy.params[:m]), 2) ?
                       defineOrder(div(length(marg_xy.params[:m]), 2)) : order
    my = marg_xy.params[:m][1:order]
    mx = marg_xy.params[:m][order+1:end]
    Vy = marg_xy.params[:v][1:order,1:order]
    Vx = marg_xy.params[:v][order+1:end, order+1:end]
    Vxy = marg_xy.params[:v][order+1:end,1:order]
    mγ = unsafeMean(marg_γ)
    D = mγ*(Vx + mx*mx')
    u = zeros(order); u[1] = mγ
    mθ = inv(D)*(Vxy + mx*my')*u
    Vθ = inv(D)
    # Check if parameters are of Array type
    Vθ = isa(Vθ, Array) ? Vθ : [Vθ]
    mθ = isa(mθ, Array) ? mθ : [mθ]
    Message(Multivariate, GaussianMeanVariance, m=mθ, v=Vθ)
end

function ruleSVariationalARIn3PPPN(marg_xy :: ProbabilityDistribution{Multivariate},
                                   marg_θ :: ProbabilityDistribution{Multivariate},
                                   marg_γ :: Nothing)
    mθ = unsafeMean(marg_θ)
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order
    mA = S+c*mθ'
    Vθ = unsafeCov(marg_θ)
    my = marg_xy.params[:m][1:order]
    mx = marg_xy.params[:m][order+1:end]
    Vy = marg_xy.params[:v][1:order,1:order]
    Vx = marg_xy.params[:v][order+1:end, order+1:end]
    Vxy = marg_xy.params[:v][order+1:end,1:order]
    B = (Vy + my*my')[1, 1] + 2*(mA*(Vxy + mx*my'))[1, 1] + (mA*(Vx + mx*mx')*mA')[1, 1] + (Vθ*(Vx + mx*mx'))[1, 1]
    Message(Gamma, a=3/2, b=B)
end

function ruleMGaussianMeanVarianceGGGD(msg_y::Message{F1, V},
                                       msg_x::Message{F2, V},
                                       dist_θ::ProbabilityDistribution,
                                       dist_γ::ProbabilityDistribution) where {F1<:Gaussian, F2<:Gaussian, V<:VariateType}

    mθ = unsafeMean(dist_θ)
    mA = S+c*mθ'
    mγ = unsafeMean(dist_γ)
    Vθ = unsafeCov(dist_θ)
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order
    trans = transition(mγ, order)
    mW = wMatrix(mγ, order)

    b_my = unsafeMean(msg_y.dist)
    b_Vy = unsafeCov(msg_y.dist)
    f_mx = unsafeMean(msg_x.dist)
    f_Vx = unsafeCov(msg_x.dist)
    D = inv(f_Vx) + mγ*Vθ
    W = [inv(b_Vy)+mW -mW*mA; -mA'*mW D+mA'*mW*mA]
    m = inv(W)*[inv(b_Vy)*b_my; inv(f_Vx)*f_mx]
    return ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=m, v=inv(W))
end
