using ForneyLab
import ARdata: use_data, generate_data
import LinearAlgebra.I, LinearAlgebra.Symmetric
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision
# order of AR model
ARorder = UInt16(10)
diagAR(dim) = Matrix{Float64}(I, dim, dim)
#coefs, x = generate_data(UInt64(10000), ARorder, 1)
# Observations
x = use_data("/Users/albertpod/Documents/Julia/VariationalBayes/data/daily-minimum-temperatures.csv", ARorder)
x = [reverse(x) for x in x]

#y = [xi[1] for xi in x[2:end]] .+ rand()*1.0
y = [x .+ 0.001 for x in x]
#push!(y, xi)

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
@RV n ~ GaussianMeanVariance(0*rand(ARorder), 0.001*diagAR(ARorder))
@RV y_t = x_t + n
#@RV n ~ GaussianMeanVariance(0.0, 0.001)
#c = zeros(ARorder); c[1] = 1.0
#@RV y_t = dot(c, x_t) + n

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
#placeholder(y_t, :y_t)
placeholder(y_t, :y_t, dims=(ARorder,))

#ForneyLab.draw(g)

# Specify recognition factorization
q = RecognitionFactorization(a, x_t_prev, x_t, w, ids=[:A, :X_t_prev, :X_t, :W])

# Inspect the subgraph for A
# ForneyLab.draw(q.recognition_factors[:A])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)

# Load algorithms
eval(Meta.parse(algo))
display(Meta.parse(algo))

# Define values for prior statistics
m_x_prev_0 = 100.0*rand(ARorder)
w_x_prev_0 = (tiny*diagAR(ARorder))
m_x_0 = 100.0*rand(ARorder)
w_x_0 = (tiny*diagAR(ARorder))
a_w_0 = 20
b_w_0 = 10
m_a_0 =  10.0*rand(ARorder)#[0.8068730972003983, 0.1686530319145092]
w_a_0 = (1*diagAR(ARorder))#[6.9639e5   6.81754e5; 6.81754e5  7.03203e5]#(1*diagAR(ARorder))

X = Vector{Float64}(undef, length(x))

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
n_its = 10
datasetRatio = 30

for t = 1:length(y)
    println("Observation # ", t)
    marginals[:a] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_a_t_min, w=w_a_t_min)
    marginals[:x_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_prev_min, w=w_x_t_prev_min)
    marginals[:x_t] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_min, w=w_x_t_min)
    marginals[:w] = ProbabilityDistribution(Univariate, Gamma, a=a_w_t_min, b=b_w_t_min)
    global m_x_t_prev_min, w_x_t_prev_min, m_x_t_min, w_x_t_min,
           a_w_t_min, b_w_t_min, m_a_t_min, w_a_t_min
    for i = 1:n_its
        #println("Iteration # ", i)


        data = Dict(:y_t   => y[t],
                    :m_a_t => m_a_t_min,
                    :w_a_t => w_a_t_min,
                    :m_x_t_prev => m_x_t_prev_min,
                    :w_x_t_prev => w_x_t_prev_min,
                    :m_x_t => m_x_t_min,
                    :w_x_t => w_x_t_min,
                    :a_w_t => a_w_t_min,
                    :b_w_t => b_w_t_min)
        #display(data)
        stepA!(data, marginals)
        stepW!(data, marginals)
        stepX_t!(data, marginals)
        stepX_t_prev!(data, marginals)
        # Extract posterior statistics
        #display(marginals[:a].params[:xi])
        #print(typeof(marginals[:a]))

        #display(mean(marginals[:x_t_min]))
        #μ, Σ = marginals[:a].params[:xi], marginals[:a].params[:w]
        #display(unsafeCov(marginals[:x_t]))

    end
    m_a[t] = unsafeMean(marginals[:a])
    w_a[t] = unsafePrecision(marginals[:a])
    m_x[t] = unsafeMean(marginals[:x_t])
    w_x[t] = unsafePrecision(marginals[:x_t])
    m_x_prev[t] = unsafeMean(marginals[:x_t_prev])
    w_x_prev[t] = unsafePrecision(marginals[:x_t_prev])
    a_w[t] = marginals[:w].params[:a]
    b_w[t] = marginals[:w].params[:b]
    # Store to buffer
    m_a_t_min = m_a[t]
    w_a_t_min = w_a[t]
    m_x_t_prev_min = m_x_prev[t]
    w_x_t_prev_min = w_x_prev[t]
    m_x_t_min = m_x[t]
    w_x_t_min = w_x[t]
    a_w_t_min = a_w[t]
    b_w_t_min = b_w[t]
    #display(marginals)
end

from = length(x) - 1000
using Plots
#predicted = [ForneyLab.sample(marginals[:a])'ForneyLab.sample(marginals[:x_t_prev]) + rand()*var(marginals[:w]) for x in x[from:end]]
#predicted = [coefs'x + rand() for x in x[from-1:end-1]]
#predicted = [x[1] for x in X[from:end]]
predicted = [x[1] for x in m_x[1:end]]
noise = [y[1] for y in y[1:end]]
actual = [x[1] for x in x[1:end]]
mse = (sum((predicted - actual).^2))/length(predicted)
#plot([actual, noise])

v_x = [v_x[1]^-1 for v_x in w_x]

# plot([predicted[1:100], predicted[1:100]], fillrange = [[m_x[1]-sqrt(w_x[1]^-1), m_x[1]+sqrt(w_x[1]^-1)] for (m_x, w_x) in zip(m_x[1:100], w_x[1:100])],
#      fillalpha = 0.2,
#      fillcolor = :red)

upto = length(x)
scale = 100
plot([predicted[1:upto], predicted[1:upto]], fillrange=[predicted[1:upto]- scale .* sqrt.(v_x[1:upto]),predicted[1:upto]+ scale .* sqrt.(v_x[1:upto])],
     fillalpha = 0.2,
     fillcolor = :red)
plot!(actual[1:upto])

# using PyPlot
#
# n_samples = 10
# from = length(x) - n_samples
#
# plot(collect(1:n_samples+1), y[from-1:end], "b*", label="y")
# plot(collect(1:n_samples+1), x[from:end], "k--", label="true x")
# plot(collect(1:n_samples), m_x[from:end-1], "b-", label="estimated x")
# #fill_between(collect(1:n_samples), m_x-sqrt.(w_x), m_x+sqrt.(w_x), color="b", alpha=0.3);
# grid("on")
# xlabel("t")
# legend(loc="upper left");
