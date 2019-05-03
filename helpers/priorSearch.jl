# Tested only with AR(1)

using ProgressMeter
using Revise
using ForneyLab
include("../AR-node/autoregression.jl")
include("../AR-node/rules_prototypes.jl")
include("../AR-node/vmp_rules.jl")

include("../helpers/functions.jl")
include("../helpers/GA.jl")
using Main.GA

include("../data/ARdata.jl")
import Main.ARdata: use_data, generate_data
import LinearAlgebra.I, LinearAlgebra.Symmetric
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision

function fitVMP(ind, len=999)
    m_x_prev_0 = ind.genes[1] .+ zeros(ARorder)
    w_x_prev_0 = (ind.genes[2]*diageye(ARorder))
    a_w_0 = ind.genes[3]
    b_w_0 = ind.genes[4]
    m_a_0 = ind.genes[5] .+ zeros(ARorder)
    w_a_0 = (ind.genes[6]*diageye(ARorder))

    m_x_prev = Vector{Vector{Float64}}(undef, len)
    w_x_prev = Vector{Array{Float64, 2}}(undef, len)
    a_w = Vector{Float64}(undef, len)
    b_w = Vector{Float64}(undef, len)
    m_a = Vector{Vector{Float64}}(undef, len)
    w_a = Vector{Array{Float64, 2}}(undef, len)

    m_x_t_prev_min = m_x_prev_0
    w_x_t_prev_min = w_x_prev_0
    a_w_t_min = a_w_0
    b_w_t_min = b_w_0
    m_a_t_min = m_a_0
    w_a_t_min = w_a_0

    marginals = Dict()
    n_its = 10
    p = Progress(length(y), 1, "Observed ")
    for t = 1:len
        update!(p, t)
        marginals[:a] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_a_t_min, w=w_a_t_min)
        marginals[:x_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_prev_min, w=w_x_t_prev_min)
        marginals[:w] = ProbabilityDistribution(Univariate, Gamma, a=a_w_t_min, b=b_w_t_min)
        for i = 1:n_its
            data = Dict(:y_t   => y[t],
                        :m_a_t => m_a_t_min,
                        :w_a_t => w_a_t_min,
                        :a_w_t => a_w_t_min,
                        :b_w_t => b_w_t_min,
                        :m_x_t_prev => m_x_t_prev_min,
                        :w_x_t_prev => w_x_t_prev_min)

            stepX_t!(data, marginals)
            stepA!(data, marginals)
            stepW!(data, marginals)
            m_a[t] = unsafeMean(marginals[:a])
            w_a[t] = unsafePrecision(marginals[:a])
            m_x_prev[t] = unsafeMean(marginals[:x_t])
            w_x_prev[t] = unsafePrecision(marginals[:x_t])
            a_w[t] = marginals[:w].params[:a]
            b_w[t] = marginals[:w].params[:b]
            # Store to buffer
            m_a_t_min = m_a[t]
            w_a_t_min = w_a[t]
            m_x_t_prev_min = m_x_prev[t]
            w_x_t_prev_min = w_x_prev[t]
            a_w_t_min = a_w[t]
            b_w_t_min = b_w[t]
        end
    end
    from = 1
    predicted = [x[1] for x in m_x_prev[from:end]]
    actual = [x[1] for x in x[from + 1:end]]
    noise = [y[1] for y in y[from:end]]
    fit_val = mse(predicted, actual)
    println("Δ(predicted, actual)=", mse(predicted, actual))
    println("Δ(predicted, noise)=", mse(predicted, noise))
    println("Δ(noise, actual)=", mse(noise, actual))
    # penalty for fitting noise
    if  mse(predicted, noise) < fit_val
        return fit_val + (fit_val - mse(predicted, noise))
    end
    return fit_val
end

# order of AR model
ARorder = 1
diageye(dim) = Matrix{Float64}(I, dim, dim)

# synthesize AR data
coefs, x = generate_data(1000, ARorder, 1.0, noise_variance=1.0)


# Observations
measurement_noise = 2.0
y = [xi[1] + sqrt(measurement_noise)*randn() for xi in x[2:end]]

# Building the model
g = FactorGraph()

# declare priors as random variables
@RV m_x_t_prev
@RV w_x_t_prev
@RV a_w_t
@RV b_w_t
@RV m_a_t
@RV w_a_t

@RV a ~ GaussianMeanPrecision(m_a_t, w_a_t)
@RV x_t_prev ~ GaussianMeanPrecision(m_x_t_prev, w_x_t_prev)
@RV w ~ Gamma(a_w_t, b_w_t)
@RV x_t = AR(a, x_t_prev, w)
@RV n ~ GaussianMeanPrecision(0.0, measurement_noise^-1)
c = zeros(ARorder); c[1] = 1.0
@RV y_t = dot(c, x_t) + n

# Placeholders for prior
placeholder(m_x_t_prev, :m_x_t_prev, dims=(ARorder,))
placeholder(w_x_t_prev, :w_x_t_prev, dims=(ARorder, ARorder))
placeholder(a_w_t, :a_w_t)
placeholder(b_w_t, :b_w_t)
placeholder(m_a_t, :m_a_t, dims=(ARorder,))
placeholder(w_a_t, :w_a_t, dims=(ARorder, ARorder))

# Placeholder for data
placeholder(y_t, :y_t)

# Specify recognition factorization
q = RecognitionFactorization(a, x_t, w, ids=[:A, :X_t, :W])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)

# Load algorithms
eval(Meta.parse(algo))

pop_size = 100

population = [GA.generate([2,3,4,6], 6) for _ in 1:pop_size]

result = GA.evolution(population, fitVMP, 100)

best = result[2][result[1][2]].genes
