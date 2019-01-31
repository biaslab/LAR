using CSV, DataFrames
using Revise
using ForneyLab

import LinearAlgebra.I, LinearAlgebra.Symmetric
# order of AR model
ARorder = 10
c = uvector(ARorder)

# Data processing
ARparse(x) = try
    parse(Float64, x)
    return true
catch
    return false
end

diagAR(dim) = Matrix{Float64}(I, dim, dim)

df = CSV.File("/Users/albertpod/Documents/Julia/Variational Bayes//daily-minimum-temperatures.csv") |> DataFrame
names!(df, [:Date, :Temp])
filter!(row -> ARparse(row[:Temp]) == true, df)

# Data
x = []
for i in range(1, size(df, 1) - ARorder)
    xi = map(x->parse(Float64,x), df[i:ARorder+i - 1, :Temp])
    push!(x, xi)
end

# Observations
# y = [xi[end] for xi in x]
y = x

# Building the model
g = FactorGraph()

# declare priors as random variables
@RV m_a
@RV w_a
@RV m_x_t_min
@RV w_x_t_min
@RV a_w
@RV b_w

@RV a ~ GaussianMeanPrecision(m_a, w_a)
@RV x_t_min ~ GaussianMeanPrecision(m_x_t_min, w_x_t_min)
#@RV x_t ~ GaussianMeanPrecision(m_x_t, w_x_t)
@RV w ~ Gamma(a_w, b_w)
@RV x_t
Autoregression(x_t, a, x_t_min, w)
@RV y_t = zeros(ARorder) + x_t
#@RV y_t
#DotProduct(y_t, c, x_t)

# Placeholders for prior
placeholder(m_a, :m_a, dims=(ARorder,))
placeholder(w_a, :w_a, dims=(ARorder, ARorder))
placeholder(m_x_t_min, :m_x_t_min, dims=(ARorder,))
placeholder(w_x_t_min, :w_x_t_min, dims=(ARorder, ARorder))
#placeholder(m_x_t, :m_x_t, dims=(ARorder,))
#placeholder(w_x_t, :w_x_t, dims=(ARorder, ARorder))
#placeholder(x_t, :x_t)
placeholder(a_w, :a_w)
placeholder(b_w, :b_w)

# Placeholder for data
placeholder(y_t, :y_t, dims=(ARorder,))
#placeholder(x_t, :x_t, dims=(ARorder,))

ForneyLab.draw(g)

# Specify recognition factorization
q = RecognitionFactorization(x_t, a, x_t_min, w, ids=[:X_t, :A, :X_t_min, :W])

# Inspect the subgraph for A
#ForneyLab.draw(q.recognition_factors[:A])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)

# Load algorithms
eval(Meta.parse(algo))

# Define values for prior statistics
m_a_0 = 0*rand(ARorder)
w_a_0 = (1*diagAR(ARorder))^-1
m_x_0 = 20*rand(ARorder)
w_x_0 = (1*diagAR(ARorder))^-1
a_w_0 = 1
b_w_0 = 5

m_a = Vector{Vector{Float64}}(undef, length(x))
w_a = Vector{Array{Float64, 2}}(undef, length(x))
m_x = Vector{Vector{Float64}}(undef, length(x))
w_x = Vector{Array{Float64, 2}}(undef, length(x))
a_w = Vector{Float64}(undef, length(x))
b_w = Vector{Float64}(undef, length(x))

m_a_min = m_a_0
w_a_min = w_a_0
m_x_t_min = m_x_0
w_x_t_min = w_x_0
a_w_min = a_w_0
b_w_min = b_w_0

marginals = Dict()
n_its = 20
datasetRatio = 3

for t = 1:div(length(y), datasetRatio)
    println("Observation # ", t)
    marginals[:a] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_a_min, w=w_a_min)
    marginals[:x_t_min] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_min, w=w_x_t_min)
    marginals[:x_t] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_min, w=w_x_t_min)
    marginals[:w] = ProbabilityDistribution(Univariate, Gamma, a=a_w_min, b=b_w_min)
    for i = 1:n_its
        println("Iteration # ", i)
        global m_a_min, w_a_min, m_x_t_min, w_x_t_min, a_w_min, b_w_min
        data = Dict(:y_t       => y[t],
                    :m_a => m_a_min,
                    :w_a => w_a_min,
                    :m_x_t_min => m_x_t_min,
                    :w_x_t_min => w_x_t_min,
                    :a_w => a_w_min,
                    :b_w => b_w_min)
        #display(data)
        stepA!(data, marginals)
        stepW!(data, marginals)
        stepX_t!(data, marginals)
        stepX_t_min!(data, marginals)
        # Extract posterior statistics
        #display(marginals[:a].params[:xi])
        #print(typeof(marginals[:a]))
        m_a[t] = marginals[:a].params[:xi]
        w_a[t] = (Symmetric(marginals[:a].params[:w]))^-1
        #display(mean(marginals[:x_t_min]))
        #μ, Σ = marginals[:a].params[:xi], marginals[:a].params[:w]
        m_x[t] = marginals[:x_t_min].params[:xi]
        w_x[t] = (Symmetric(marginals[:x_t_min].params[:w]))^-1
        #mxt[t] = mean(marginals[:x_t])
        #wxt[t] = cov(marginals[:x_t])^-1
        a_w[t] = mean(marginals[:w])
        b_w[t] = (var(marginals[:w]))^-1
        # Store to buffer
        m_a_min = m_a[t]
        w_a_min = w_a[t]
        #m_x_t_min = mxt[t]
        #w_x_t_min = wxt[t]
        m_x_t_min = m_x[t]
        w_x_t_min = w_x[t]
        a_w_min = a_w[t]
        b_w_min = b_w[t]
    end
end
