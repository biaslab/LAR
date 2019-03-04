using ForneyLab
import ARdata: daily_temperature, generate_data
import LinearAlgebra.I, LinearAlgebra.Symmetric
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision
using Plots


function plotter(ma, x, mw, testPoints=100, plt=true)
    from = length(x) - testPoints
    predicted = [ForneyLab.sample(marginals[:a])'x + var(marginals[:w]) for x in x[from-1:end-1]]
    actual = [x[1] for x in x[from:end]]
    mse = (sum((predicted - actual).^2))/length(predicted)
    if plt
        ylims!(-10, 30)
        plot!([actual, predicted], title = "unforeseen data", xlabel="days", ylabel="temperature", label=["actual", "predicted"])
    end
    return mse
end

# order of AR model
ARorder = UInt16(10)
diagAR(dim) = Matrix{Float64}(I, dim, dim)
x = []
# Observations
x = daily_temperature("/Users/albertpod/Documents/Julia/VariationalBayes/data/daily-minimum-temperatures.csv", ARorder)
x = [reverse(x) for x in x]
#append!(x, x); append!(x, x)
#y = x
#plot([x[1] for x in x])
#plot([x[1] for x in x][end - 100: end])
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
@RV x_t ~ GaussianMeanPrecision(m_x_t, w_x_t)
#@RV x_t
@RV w ~ Gamma(a_w_t, b_w_t)
Autoregression(x_t, x_t_prev, a, w)
#@RV n ~ GaussianMeanPrecision(zeros(ARorder), zeros(ARorder, ARorder))
#@RV y_t = zeros(ARorder) + x_t
#c = zeros(ARorder); c[1] = 1.0
#@RV y_t = dot(x_t, c) + n
#GainEquality(y_t, x_t_prev, x_t, c)

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
#placeholder(y_t, :y_t, dims=(ARorder,))
#placeholder(x_t, :x_t, dims=(ARorder,))

ForneyLab.draw(g)

# Specify recognition factorization
q = RecognitionFactorization(a, w, ids=[:A, :W])

# Inspect the subgraph for A
# ForneyLab.draw(q.recognition_factors[:A])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)

# Load algorithms
eval(Meta.parse(algo))
display(Meta.parse(algo))

# Define values for prior statistics
m_x_prev_0 = x[1]#10*rand(ARorder)
w_x_prev_0 = (huge*diagAR(ARorder))
m_x_0 = x[2]
w_x_0 = (huge*diagAR(ARorder))
a_w_0 = 20
b_w_0 = 2
m_a_0 = 5.0*rand(ARorder)
w_a_0 = (0.25*diagAR(ARorder))

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
testPoints = 100

MSEs = []
anim = @animate for t = 2:length(x)-testPoints
    s = plot()
    println("Observation # ", t)
    marginals[:a] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_a_t_min, w=w_a_t_min)
    marginals[:x_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_prev_min, w=w_x_t_prev_min)
    marginals[:x_t] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_min, w=w_x_t_min)
    marginals[:w] = ProbabilityDistribution(Univariate, Gamma, a=a_w_t_min, b=b_w_t_min)
    for i = 1:n_its
        #println("Iteration # ", i)
        global m_x_t_prev_min, w_x_t_prev_min, m_x_t_min, w_x_t_min,
               a_w_t_min, b_w_t_min, m_a_t_min, w_a_t_min

        data = Dict(:m_a_t => m_a_t_min,
                    :w_a_t => w_a_t_min,
                    :m_x_t_prev => m_x_t_prev_min,
                    :w_x_t_prev => w_x_t_prev_min,
                    :m_x_t => m_x_t_min,
                    :w_x_t => w_x_t_min,
                    :a_w_t => a_w_t_min,
                    :b_w_t => b_w_t_min)
        stepA!(data, marginals)
        stepW!(data, marginals)
        m_a[t] = unsafeMean(marginals[:a])
        w_a[t] = unsafePrecision(marginals[:a])
        m_x[t] = unsafeMean(marginals[:x_t])
        w_x[t] = (huge*diagAR(ARorder))
        m_x_prev[t] = unsafeMean(marginals[:x_t_prev])
        w_x_prev[t] = (huge*diagAR(ARorder))
        a_w[t] = marginals[:w].params[:a]
        b_w[t] = marginals[:w].params[:b]
        m_a_t_min = m_a[t]
        w_a_t_min = w_a[t]
        m_x_t_prev_min = x[t]
        w_x_t_prev_min = w_x_prev[t]
        m_x_t_min = x[t+1]
        w_x_t_min = w_x[t]
        a_w_t_min = a_w[t]
        b_w_t_min = b_w[t]
    end
    mse = analyse(marginals[:a], x, marginals[:w], testPoints, true)
    display(mse)
    push!(MSEs, mse)
end

gif(anim, "gifs/AR-online.gif", fps = 10)

init = ProbabilityDistribution(Multivariate, GaussianMeanPrecision,
                               m=10.0*rand(ARorder),
                               w=(tiny*diagAR(ARorder)))
init_predicted = [ForneyLab.sample(init)'x + (20/2)^-1 for x in x[from-1:end-1]]
mse_init = (sum((init_predicted - actual).^2))/length(init_predicted)

plot([actual, predicted], xlabel="timestamp", ylabel = "x", label=["actual", "predicted"])
