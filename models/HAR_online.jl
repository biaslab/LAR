using ProgressMeter
using Revise
using ForneyLab
include("../AR-node/autoregression.jl")
include("../AR-node/rules_prototypes.jl")
include("../AR-node/vmp_rules.jl")

include("../helpers.jl")

include("../data/ARdata.jl")
import Main.ARdata: use_data, generate_data
import LinearAlgebra.I, LinearAlgebra.Symmetric
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision

# order of AR model
ARorder = 10
diagAR(dim) = Matrix{Float64}(I, dim, dim)

# synthesize AR data
coefs, x = generate_data(10000, ARorder, 1.0, noise_variance=0.1)

# daily temperature
#x = use_data("data/daily-minimum-temperatures.csv", ARorder)
#x = [reverse(x) for x in x]

# Observations
measurement_noise = 1.0
y = [xi[1] + sqrt(measurement_noise)*randn() for xi in x[2:end]]


# Building the model
g = FactorGraph()

# declare priors as random variables
@RV m_x_t_prev
@RV w_x_t_prev
@RV m_x_t
@RV w_x_t
@RV a_w_t
@RV b_w_t
@RV m_a_t
@RV w_a_t

@RV a ~ GaussianMeanPrecision(m_a_t, w_a_t)
@RV x_t_prev ~ GaussianMeanPrecision(m_x_t_prev, w_x_t_prev)
@RV w ~ Gamma(a_w_t, b_w_t)
@RV x_t ~ GaussianMeanPrecision(m_x_t, w_x_t)
Autoregression(x_t, x_t_prev, a, w)
@RV n ~ GaussianMeanPrecision(0.0, 1.0)
c = zeros(ARorder); c[1] = 1.0
@RV y_t = dot(c, x_t) + n

# Placeholders for prior
placeholder(m_x_t_prev, :m_x_t_prev, dims=(ARorder,))
placeholder(w_x_t_prev, :w_x_t_prev, dims=(ARorder, ARorder))
placeholder(m_x_t, :m_x_t, dims=(ARorder,))
placeholder(w_x_t, :w_x_t, dims=(ARorder, ARorder))
placeholder(a_w_t, :a_w_t)
placeholder(b_w_t, :b_w_t)
placeholder(m_a_t, :m_a_t, dims=(ARorder,))
placeholder(w_a_t, :w_a_t, dims=(ARorder, ARorder))

# Placeholder for data
placeholder(y_t, :y_t)

#ForneyLab.draw(g)

# Specify recognition factorization
q = RecognitionFactorization(a, x_t_prev, x_t, w, ids=[:A, :X_t_prev, :X_t, :W])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)

# Load algorithms
eval(Meta.parse(algo))

# Define values for prior statistics
m_x_prev_0 = 0.0*rand(ARorder)
w_x_prev_0 = (tiny*diagAR(ARorder))
m_x_0 = 0.0*rand(ARorder)
w_x_0 = (tiny*diagAR(ARorder))
a_w_0 = tiny
b_w_0 = huge
m_a_0 =  1.0*rand(ARorder)
w_a_0 =  (tiny*diagAR(ARorder))

m_x_prev = Vector{Vector{Float64}}(undef, length(x))
w_x_prev = Vector{Array{Float64, 2}}(undef, length(x))
m_x = Vector{Vector{Float64}}(undef, length(x))
w_x = Vector{Array{Float64, 2}}(undef, length(x))
a_w = Vector{Float64}(undef, length(x))
b_w = Vector{Float64}(undef, length(x))
m_a = Vector{Vector{Float64}}(undef, length(x))
w_a = Vector{Array{Float64, 2}}(undef, length(x))

m_x_t_prev_min = m_x_prev_0
w_x_t_prev_min = w_x_prev_0
m_x_t_min = m_x_0
w_x_t_min = w_x_0
a_w_t_min = a_w_0
b_w_t_min = b_w_0
m_a_t_min = m_a_0
w_a_t_min = w_a_0

marginals = Dict()
n_its = 5
datasetRatio = 30

p = Progress(length(y), 1, "Observed ")
for t = 1:length(y)
    update!(p, t)
    marginals[:a] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_a_t_min, w=w_a_t_min)
    marginals[:x_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_prev_min, w=w_x_t_prev_min)
    marginals[:x_t] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_min, w=w_x_t_min)
    marginals[:w] = ProbabilityDistribution(Univariate, Gamma, a=a_w_t_min, b=b_w_t_min)
    global m_x_t_prev_min, w_x_t_prev_min, m_x_t_min, w_x_t_min,
           a_w_t_min, b_w_t_min, m_a_t_min, w_a_t_min

    for i = 1:n_its
        data = Dict(:y_t   => y[t],
                    :m_a_t => m_a_t_min,
                    :w_a_t => w_a_t_min,
                    :m_x_t_prev => m_x_t_prev_min,
                    :w_x_t_prev => w_x_t_prev_min,
                    :m_x_t => m_x_t_min,
                    :w_x_t => w_x_t_min,
                    :a_w_t => a_w_t_min,
                    :b_w_t => b_w_t_min)

        stepA!(data, marginals)
        stepW!(data, marginals)
        stepX_t!(data, marginals)
        stepX_t_prev!(data, marginals)
    end

    m_a[t] = unsafeMean(marginals[:a])
    w_a[t] = unsafePrecision(marginals[:a])
    m_x[t] = unsafeMean(marginals[:x_t])
    w_x[t] = unsafePrecision(marginals[:x_t])
    m_x_prev[t] = unsafeMean(marginals[:x_t_prev])
    w_x_prev[t] = unsafePrecision(marginals[:x_t_prev])
    a_w[t] = a_w_0#marginals[:w].params[:a]
    b_w[t] = b_w_0#marginals[:w].params[:b]
    # Store to buffer
    m_a_t_min = m_a[t]
    w_a_t_min = w_a[t]
    m_x_t_prev_min = m_x_prev[t]
    w_x_t_prev_min = w_x_prev[t]
    m_x_t_min = m_x[t]
    w_x_t_min = w_x[t]
    a_w_t_min = a_w[t]
    b_w_t_min = b_w[t]
end

predicted = [x[1] for (t, x) in enumerate(m_x[1:end-1])]
noise = [y[1] for y in y[1:end]]
actual = [x[1] for x in x[2:end]]
println("Δ(predicted, noise)=", mse(predicted, noise))
println("Δ(predicted, actual)=", mse(predicted, actual))

v_x = [v_x[1]^-1 for v_x in w_x[1:end-1]] # variances of estimated state

using Plots
upto = 100 # limit for building a graph
scale = 1.0 # scale for the variance
plot([predicted[1:upto], predicted[1:upto]], fillrange=[predicted[1:upto] -
      scale .* sqrt.(v_x[1:upto]), predicted[1:upto] +
      scale .* sqrt.(v_x[1:upto])],
      fillalpha = 0.2,
      fillcolor = :red,
      label=["inferred", "inferred"])
plot!(noise[1:upto], label="noised")
plot!(actual[1:upto], label="real state")
