# joint estimations of x and a with known process and measurement noises

using ProgressMeter
using Revise
using ForneyLab
using Random
using Plots
include( "../AR-node/autoregression.jl")
include( "../AR-node/observationAR.jl")
include("../AR-node/rules_prototypes.jl")
include("../AR-node/vmp_rules.jl")
include("../helpers/functions.jl")
include("../data/ARdata.jl")
import Main.ARdata: generateHAR
import LinearAlgebra.I, LinearAlgebra.Symmetric
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision

Random.seed!(42)

ARorder = 2

data = generateHAR(1000, ARorder, levels=2, nvars=[0.1, 0.5])
x = [x[1] for x in data[3]]
# Observations
v_y = 1.0
y = [xi[1] + sqrt(v_y)*randn() for xi in data[3]]

# Creating the graph
g = FactorGraph()

# First layer
@RV m_θ1
@RV w_θ1
@RV θ1 ~ GaussianMeanPrecision(m_θ1, w_θ1)
@RV a_θ2
@RV b_θ2
@RV w_θ2 ~ Gamma(a_θ2, b_θ2)
@RV m_θ2_t_prev
@RV w_θ2_t_prev
@RV θ2_t_prev ~ GaussianMeanPrecision(m_θ2_t_prev, w_θ2_t_prev)
@RV θ2_t = AR(θ1, θ2_t_prev, w_θ2)

# Second layer
@RV a_x
@RV b_x
@RV w_x ~ Gamma(a_x, b_x)
@RV m_x_t_prev
@RV w_x_t_prev
@RV x_t_prev ~ GaussianMeanPrecision(m_x_t_prev, w_x_t_prev)
@RV x_t = AR(θ2_t, x_t_prev, w_x)

# Observation
@RV m_y_t
@RV w_y_t
observationAR(m_y_t, x_t, w_y_t)

# Placeholders for prior of upper layer
placeholder(m_θ1, :m_θ1, dims=(ARorder,))
placeholder(w_θ1, :w_θ1, dims=(ARorder, ARorder))
placeholder(a_θ2, :a_θ2)
placeholder(b_θ2, :b_θ2)
placeholder(m_θ2_t_prev, :m_θ2_t_prev, dims=(ARorder,))
placeholder(w_θ2_t_prev, :w_θ2_t_prev, dims=(ARorder, ARorder))

# Placeholders for prior of bottom layer
placeholder(a_x, :a_x)
placeholder(b_x, :b_x)
placeholder(m_x_t_prev, :m_x_t_prev, dims=(ARorder,))
placeholder(w_x_t_prev, :w_x_t_prev, dims=(ARorder, ARorder))

# Placeholder for observations
placeholder(m_y_t, :m_y_t)
placeholder(w_y_t, :w_y_t)

ForneyLab.draw(g)

# Specify recognition factorization
q = RecognitionFactorization(θ2_t, θ1, θ2_t_prev, w_θ2, x_t, x_t_prev, w_x,
                            ids=[:θ2_t :θ1 :θ2_t_prev :W_θ2 :X_t :X_t_prev :W_x])

algo = variationalAlgorithm(q)
algoF = freeEnergyAlgorithm(q)

# Load algorithms
eval(Meta.parse(algo))
eval(Meta.parse(algoF))


# Storage for upper layer
m_θ2 = Vector{Vector{Float64}}(undef, length(y))
w_θ2 = Vector{Array{Float64, 2}}(undef, length(y))
m_θ1 = Vector{Vector{Float64}}(undef, length(y))
w_θ1 = Vector{Array{Float64, 2}}(undef, length(y))
a_θ2 = Vector{Float64}(undef, length(y))
b_θ2 = Vector{Float64}(undef, length(y))

# Storage for bottom layer
m_x = Vector{Vector{Float64}}(undef, length(y))
w_x = Vector{Array{Float64, 2}}(undef, length(y))
a_x = Vector{Float64}(undef, length(y))
b_x = Vector{Float64}(undef, length(y))

# Define values for upper layer
m_θ1_0 = zeros(ARorder)
w_θ1_0 = diagAR(ARorder)
a_θ2_0 = 1
b_θ2_0 = 0.1
m_θ2_prev_0 = zeros(ARorder)
w_θ2_prev_0 = diagAR(ARorder)
m_θ2_t_0 = zeros(ARorder)
w_θ2_t_0 = diagAR(ARorder)

# Define values for bottom layer
a_x_0 = 1
b_x_0 = 0.5
m_x_t_prev_0 = zeros(ARorder)
w_x_t_prev_0 = diagAR(ARorder)

# Priors upper layer
m_θ1_min = m_θ1_0
w_θ1_min = w_θ1_0
a_θ2_min = a_θ2_0
b_θ2_min = b_θ2_0
m_θ2_t_prev_min = m_θ2_prev_0
w_θ2_t_prev_min = w_θ2_prev_0

# Priors bottom layer
a_x_min = a_x_0
b_x_min = b_x_0
m_θ2_t_min = m_θ2_t_0
w_θ2_t_min = w_θ2_t_0
m_x_t_prev_min = m_x_t_prev_0
w_x_t_prev_min = w_x_t_prev_0


data = Dict()
marginals = Dict()
n_its = 5

# Storage for scores
FAR = []

p = Progress(length(y), 1, "Observed ")
for t in 1:length(y)
    update!(p, t)
    # Upper Layer
    marginals[:θ1] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_θ1_min, w=w_θ1_min)
    marginals[:θ2_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_θ2_t_prev_min, w=w_θ2_t_prev_min)
    marginals[:θ2_t] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_θ2_t_min, w=w_θ2_t_min)
    marginals[:w_θ2] = ProbabilityDistribution(Univariate, Gamma, a=a_θ2_min, b=b_θ2_min)
    # Bottom Layer
    marginals[:x_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_prev_min, w=w_x_t_prev_min)
    marginals[:w_x] = ProbabilityDistribution(Univariate, Gamma, a=a_θ2_min, b=b_θ2_min)

    global m_θ1_min, w_θ1_min, m_θ2_t_prev_min, w_θ2_t_prev_min, a_θ2_min, b_θ2_min,
           m_x_t_prev_min, w_x_t_prev_min, a_x_min, b_x_min, data

    for i = 1:n_its
        data = Dict(:m_y_t => y[t],
                    :w_y_t => v_y^-1,
                    :m_θ1 => m_θ1_min,
                    :w_θ1 => w_θ1_min,
                    :m_θ2_t_prev => m_θ2_t_prev_min,
                    :w_θ2_t_prev => w_θ2_t_prev_min,
                    :a_θ2 => a_θ2_min,
                    :b_θ2 => b_θ2_min,
                    :m_x_t_prev => m_x_t_prev_min,
                    :w_x_t_prev => w_x_t_prev_min,
                    :a_x => a_x_min,
                    :b_x => b_x_min)

        stepX_t!(data, marginals)
        stepW_x!(data, marginals)
        stepθ2_t!(data, marginals)
        stepθ1!(data, marginals)
        stepW_θ2!(data, marginals)

        stepθ2_t_prev!(data, marginals)
        stepθ2_t!(data, marginals)

        stepX_t_prev!(data, marginals)
        stepX_t!(data, marginals)

        m_θ2[t] = unsafeMean(marginals[:θ2_t])
        w_θ2[t] = unsafePrecision(marginals[:θ2_t])
        m_θ1[t] = unsafeMean(marginals[:θ1])
        w_θ1[t] = unsafePrecision(marginals[:θ1])
        a_θ2[t] = marginals[:w_θ2].params[:a]
        b_θ2[t] = marginals[:w_θ2].params[:b]
        m_x[t] = unsafeMean(marginals[:x_t])
        w_x[t] = unsafePrecision(marginals[:x_t])
        a_x[t] = marginals[:w_x].params[:a]
        b_x[t] = marginals[:w_x].params[:b]

        m_θ1_min = m_θ1[t]
        w_θ1_min = w_θ1[t]
        m_θ2_t_prev_min = m_θ2[t]
        w_θ2_t_prev_min = w_θ2[t]
        m_x_t_prev_min = m_x[t]
        w_x_t_prev_min = w_x[t]
        a_θ2_min = a_θ2[t]
        b_θ2_min = b_θ2[t]
        a_x_min = a_x[t]
        b_x_min = b_x[t]
    end
    push!(FAR, (freeEnergy(data, marginals)))
end

from = 500; upto=550
scatter(y[from:upto], label="observed")
plot!(x[from:upto], label="real")
plot!([m_x[1] for m_x in m_x[from:upto]], label="inferred")
