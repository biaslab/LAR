using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy
import SpecialFunctions: polygamma, digamma
export Autoregressive, AR, averageEnergy, slug, differentialEntropy

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

    Autoregressive(out, θ, in, γ, id=:some_id)
"""
mutable struct Autoregressive <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function Autoregressive(y, θ, x, γ; id=generateId(Autoregressive))
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

slug(::Type{Autoregressive}) = "AR"

# Average energy functional can' be computed for AR node when copying operator is used
function averageEnergy(::Type{Autoregressive},
                       marg_y::ProbabilityDistribution{Univariate},
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
