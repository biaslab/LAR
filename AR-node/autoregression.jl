import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!, averageEnergy
import SpecialFunctions: polygamma, digamma
export Autoregression, AR, averageEnergy

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

function AR(θ::Variable, x::Variable, γ::Variable)
    y = Variable()
    Autoregression(y, θ, x, γ)
    return y
end

ForneyLab.slug(::Type{Autoregression}) = "AR"

# Average energy functional
function averageEnergy(::Type{Autoregression},
                       marg_y::ProbabilityDistribution{Multivariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_γ::ProbabilityDistribution{Univariate})
    order = length(mean(marg_y))
    mθ = unsafeMean(marg_θ)
    covθ = unsafeCov(marg_θ)
    mA = S+c*mθ'
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    mγ = unsafeMean(marg_γ)
    mW = wMatrix(mγ, order)
    Vx = unsafeCov(marg_x)
    B1 = tr(mW*unsafeCov(marg_y)) + my'*mW*my - (mA*mx)'*mW*my - my'*mW*mA*mx + tr(S'*mW*S*Vx)
    B2 = mγ*mθ'*Vx*mθ + tr(S'*mW*S*mx*mx') + mγ*mx'*covθ*mx + mγ*mθ'*mx*mx'*mθ
    -0.5*(digamma(marg_γ.params[:a]) - log(marg_γ.params[:b]) - 0.5*(1-order)*log(tiny) + 0.5*order*log(2*pi)) + 0.5*(B1 + B2)
end
