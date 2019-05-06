import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!, averageEnergy
import SpecialFunctions: polygamma
export Autoregression, AR, slug, averageEnergy

"""
Description:

    A Gaussian mixture with mean-precision parameterization:

    f(out, a, x, W) = 𝒩(out|Ax, W^-1),

    where A =    a^T
                I	0

Interfaces:

    1. out
    2. a (autoregression coefficients)
    3. x (input vector)
    4. W (precision)

Construction:

    Autoregression(out, x, a, W, id=:some_id)
"""
mutable struct Autoregression <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function Autoregression(out, a, x, w; id=generateId(Autoregression))
        @ensureVariables(out, x, a, w)
        self = new(id, Array{Interface}(undef, 4), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:a] = self.interfaces[3] = associate!(Interface(self), a)
        self.i[:W] = self.interfaces[4] = associate!(Interface(self), w)
        return self
    end
end

function AR(a::Variable, x::Variable, w::Variable)
    out = Variable()
    Autoregression(out, a, x, w)
    return out
end

slug(::Type{Autoregression}) = "AR"

# Average energy functional
function averageEnergy(::Type{Autoregression},
                       marg_y::ProbabilityDistribution{Multivariate},
                       marg_a::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_w::ProbabilityDistribution{Univariate})
    dim = length(mean(marg_y))
    ma = unsafeMean(marg_a)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    B = unsafeCov(marg_y)[1, 1] + my[1]*my'[1] - 2*my[1]*ma'*mx + ma'*(unsafeCov(marg_x)+mx*mx')*ma + mx'*unsafeCov(marg_a)*mx

    -0.5*(polygamma(0, marg_w.params[:a]) - log(marg_w.params[:b]) + (1-dim)*log(tiny) - 0.5*dim*log(2*pi)) + 0.5*mean(marg_w)*B
end
