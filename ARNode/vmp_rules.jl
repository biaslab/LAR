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
    B = (Vθ*(Vx + mx*mx'))[1, 1] + 2*(mA*(Vxy + mx*my'))[1, 1] + (mA*(Vx + mx*mx')*mA')[1, 1] + (Vy + my*my')[1, 1]
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

    f_msg_y = ruleSVariationalAROutNPPP(nothing, msg_x, dist_θ, dist_γ)
    b_msg_x = ruleSVariationalARIn1PNPP(msg_y, nothing, dist_θ, dist_γ)

    f_Vx = unsafeCov(msg_x.dist)
    f_Vy = unsafeCov(f_msg_y.dist)
    b_Vy = unsafeCov(msg_y.dist)

    marg_y = f_msg_y.dist * msg_y.dist
    marg_x = b_msg_x.dist * msg_x.dist

    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x);
    Vy = unsafeCov(marg_y)
    Vx = unsafeCov(marg_x);
    D =  inv(f_Vx) + mγ*Vθ
    Vxy = inv(D)*mA'*inv(f_Vy + b_Vy)*b_Vy

    return ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=[my; mx], v=[Vy Vxy; Vxy' Vx])
end
