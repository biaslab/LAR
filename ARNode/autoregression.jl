using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy
import SpecialFunctions: polygamma, digamma
export Autoregression, AR, averageEnergy, slug, differentialEntropy

"""
Description:

    A Gaussian mixture with mean-precision parameterization:

    f(y, θ, x, γ) = 𝒩(out|A(θ)x, V(γ)),

    where A(θ) =  θᵀ
                I	0

Interfaces:

    1. y (output vector)
    2. θ (autoregression coefficients)
    3. x (input vector)
    4. γ (precision)

Construction:

    Autoregression(out, θ, in, γ, id=:some_id)
"""
mutable struct Autoregression <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function Autoregression(y, θ, x, γ; id=generateId(Autoregression))
        @ensureVariables(y, x, θ, γ)
        self = new(id, Array{Interface}(undef, 4), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:θ] = self.interfaces[3] = associate!(Interface(self), θ)
        self.i[:γ] = self.interfaces[4] = associate!(Interface(self), γ)
        return self
    end
end

slug(::Type{Autoregression}) = "AR"

# Average energy functional
function averageEnergy(::Type{Autoregression},
                       marg_y::ProbabilityDistribution{Multivariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_γ::ProbabilityDistribution{Univariate})
    order = length(mean(marg_y))
    mθ = unsafeMean(marg_θ)
    Vθ = unsafeCov(marg_θ)
    mA = S+c*mθ'
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    mγ = unsafeMean(marg_γ)
    mW = wMatrix(mγ, order)
    Vx = unsafeCov(marg_x)
    Vy = unsafeCov(marg_y)
    B1 = tr(mW*unsafeCov(marg_y)) + my'*mW*my - (mA*mx)'*mW*my - my'*mW*mA*mx + tr(S'*mW*S*Vx)
    B2 = mγ*tr(Vθ*Vx) + mγ*mθ'*Vx*mθ + tr(S'*mW*S*mx*mx') + mγ*mx'*Vθ*mx + mγ*mθ'*mx*mx'*mθ
    valid = -0.5*(digamma(marg_γ.params[:a]) - log(marg_γ.params[:b])) + 0.5*log(2*pi) + 0.5*mγ*(Vy[1]+(my[1])^2 - 2*mθ'*mx*my[1] + tr(Vθ*Vx) + mx'*Vθ*mx + mθ'*(Vx + mx*mx')*mθ)
end

# This is really dirty, but this is the only way to compute FE for meanfield assumption now
function differentialEntropy(dist::ProbabilityDistribution{Multivariate, F}) where F<:Gaussian


    distAR = convert(ProbabilityDistribution{Multivariate, GaussianMeanVariance}, dist)
    dim = size((distAR.params[:v]))[1]
    if dim > 1 && sum(distAR.params[:v][2:dim]) < (dim-1)*tiny
        return 0.5*log(det(distAR.params[:v][1])) + (1/2)*log(2*pi) + (1/2)
    else
        return 0.5*log(det(unsafeCov(dist))) + (dims(dist)/2)*log(2*pi) + (dims(dist)/2)
    end
end
