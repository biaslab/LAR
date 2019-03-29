# joint estimations of x, a and w (process noise)

using ProgressMeter
using Revise
using ForneyLab
include( "../AR-node/autoregression.jl")
include("../AR-node/rules_prototypes.jl")
include("../AR-node/vmp_rules.jl")
include( "../AR-node/observationAR.jl")
include("../helpers/functions.jl")
include("../data/ARdata.jl")
import Main.ARdata: use_data, generate_data
import LinearAlgebra.I, LinearAlgebra.Symmetric
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision

# order of AR model
ARorder = 1
diagAR(dim) = Matrix{Float64}(I, dim, dim)
x = []

# AR data
a_w = 1.0; b_w = 1.0
process_noise = b_w/a_w
coefs, x = generate_data(100, ARorder, 1, noise_variance=process_noise)

# Observations
y = [xi[1] for xi in x[ARorder:end]]

g = FactorGraph()

# declare priors as random variables
@RV m_x_t_prev
@RV w_x_t_prev
@RV m_a_t
@RV w_a_t
@RV m_y_t
@RV w_y_t
@RV a_w_t
@RV b_w_t

@RV a ~ GaussianMeanPrecision(m_a_t, w_a_t)
@RV x_t_prev ~ GaussianMeanPrecision(m_x_t_prev, w_x_t_prev)
@RV w ~ Gamma(a_w_t, b_w_t)
@RV x_t = AR(a, x_t_prev, w)
observationAR(m_y_t, x_t, w_y_t)

# Placeholders for prior
placeholder(m_a_t, :m_a_t, dims=(ARorder,))
placeholder(w_a_t, :w_a_t, dims=(ARorder, ARorder))

# Placeholder for data
placeholder(m_x_t_prev, :m_x_t_prev, dims=(ARorder,))
placeholder(w_x_t_prev, :w_x_t_prev, dims=(ARorder, ARorder))
placeholder(a_w_t, :a_w_t)
placeholder(b_w_t, :b_w_t)
placeholder(m_y_t, :m_y_t)
placeholder(w_y_t, :w_y_t)

#ForneyLab.draw(g)

# Specify recognition factorization
q = RecognitionFactorization(a, x_t, w, ids=[:A :X_t :W])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)
algoF = freeEnergyAlgorithm(q)

# Load algorithms
eval(Meta.parse(algo))
eval(Meta.parse(algoF))


# z = huge
#
# while z < 0.9 || z > 1.2
#     # Define values for prior statistics
#     m_a_0 = 0.0*rand(ARorder)
#     w_a_0 = (tiny*diagAR(ARorder))
#     m_x_prev_0 = x[ARorder - 1]
#     w_x_prev_0 = (0.1*diagAR(ARorder))
#     a_w_0 = rand(0:0.0001:1000)
#     b_w_0 = rand(0:0.0001:1000)
#
#     m_x_prev = Vector{Vector{Float64}}(undef, length(y))
#     w_x_prev = Vector{Array{Float64, 2}}(undef, length(y))
#     m_a = Vector{Vector{Float64}}(undef, length(y))
#     w_a = Vector{Array{Float64, 2}}(undef, length(y))
#     a_w = Vector{Float64}(undef, length(y))
#     b_w = Vector{Float64}(undef, length(y))
#
#     m_x_t_prev_min = m_x_prev_0
#     w_x_t_prev_min = w_x_prev_0
#     m_a_t_min = m_a_0
#     w_a_t_min = w_a_0
#     a_w_t_min = a_w_0
#     b_w_t_min = b_w_0
#
#     marginals = Dict()
#     n_its = 1
#
#     # Storage for predictions
#     predictions = []
#     F = []
#
#     p = Progress(length(y), 1, "Observed ")
#     for t in 1:length(y)
#         #println("Observation #", t)
#         update!(p, t)
#         marginals[:a] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_a_t_min, w=w_a_t_min)
#         marginals[:x_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_prev_min, w=w_x_t_prev_min)
#         marginals[:w] = ProbabilityDistribution(Univariate, Gamma, a=a_w_t_min, b=b_w_t_min)
#         push!(predictions, m_a_t_min'm_x_t_prev_min)
#         #global m_x_t_prev_min, w_x_t_prev_min, m_a_t_min, w_a_t_min, a_w_t_min, b_w_t_min
#
#         for i = 1:n_its
#             data = Dict(:m_y_t => y[t],
#                         :w_y_t => huge,
#                         :m_a_t => m_a_t_min,
#                         :w_a_t => w_a_t_min,
#                         :m_x_t_prev => m_x_t_prev_min,
#                         :w_x_t_prev => w_x_t_prev_min,
#                         :a_w_t => a_w_t_min,
#                         :b_w_t => b_w_t_min)
#             stepX_t!(data, marginals)
#             stepA!(data, marginals)
#             stepW!(data, marginals)
#             m_a[t] = unsafeMean(marginals[:a])
#             w_a[t] = unsafePrecision(marginals[:a])
#             m_x_prev[t] = unsafeMean(marginals[:x_t])
#             w_x_prev[t] = (huge*diagAR(ARorder))
#             a_w[t] = marginals[:w].params[:a]
#             b_w[t] = marginals[:w].params[:b]
#             m_a_t_min = m_a[t]
#             w_a_t_min = w_a[t]
#             m_x_t_prev_min = m_x_prev[t]
#             w_x_t_prev_min = w_x_prev[t]
#             a_w_t_min = a_w[t]
#             b_w_t_min = b_w[t]
#             push!(F, log(abs(Complex(freeEnergy(data, marginals)))))
#         end
#     end
#     global z
#     z = mean(marginals[:w])^-1
#     display(z)
# end

# Define values for prior statistics
m_a_0 = 1.0*rand(ARorder)
w_a_0 = (0.0001*diagAR(ARorder))
m_x_prev_0 = 1.0*rand(ARorder)
w_x_prev_0 = (0.0001*diagAR(ARorder))
a_w_0 = 0.001
b_w_0 = 0.001

m_x_prev = Vector{Vector{Float64}}(undef, length(y))
w_x_prev = Vector{Array{Float64, 2}}(undef, length(y))
m_a = Vector{Vector{Float64}}(undef, length(y))
w_a = Vector{Array{Float64, 2}}(undef, length(y))
a_w = Vector{Float64}(undef, length(y))
b_w = Vector{Float64}(undef, length(y))

m_x_t_prev_min = m_x_prev_0
w_x_t_prev_min = w_x_prev_0
m_a_t_min = m_a_0
w_a_t_min = w_a_0
a_w_t_min = a_w_0
b_w_t_min = b_w_0

marginals = Dict()
n_its = 1

# Storage for predictions
predictions = []
F = []

p = Progress(length(y), 1, "Observed ")
for t in 1:length(y)
    update!(p, t)
    marginals[:a] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_a_t_min, w=w_a_t_min)
    marginals[:x_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_prev_min, w=w_x_t_prev_min)
    marginals[:w] = ProbabilityDistribution(Univariate, Gamma, a=a_w_t_min, b=b_w_t_min)
    push!(predictions, m_a_t_min'm_x_t_prev_min)
    global m_x_t_prev_min, w_x_t_prev_min, m_a_t_min, w_a_t_min, a_w_t_min, b_w_t_min

    for i = 1:n_its
        data = Dict(:m_y_t => y[t],
                    :w_y_t => huge,
                    :m_a_t => m_a_t_min,
                    :w_a_t => w_a_t_min,
                    :m_x_t_prev => m_x_t_prev_min,
                    :w_x_t_prev => w_x_t_prev_min,
                    :a_w_t => a_w_t_min,
                    :b_w_t => b_w_t_min)
        stepX_t!(data, marginals)
        stepA!(data, marginals)
        stepW!(data, marginals)
        m_a[t] = unsafeMean(marginals[:a])
        w_a[t] = unsafePrecision(marginals[:a])
        m_x_prev[t] = unsafeMean(marginals[:x_t])
        w_x_prev[t] = unsafePrecision(marginals[:x_t])
        a_w[t] = marginals[:w].params[:a]
        b_w[t] = marginals[:w].params[:b]
        m_a_t_min = m_a[t]
        w_a_t_min = w_a[t]
        m_x_t_prev_min = m_x_prev[t]
        w_x_t_prev_min = w_x_prev[t]
        a_w_t_min = a_w[t]
        b_w_t_min = b_w[t]
        push!(F, freeEnergy(data, marginals))
    end
end
