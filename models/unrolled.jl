# Offline learning

using ProgressMeter
using Revise
using Random
using ForneyLab
using LaTeXStrings
include("../data/ARdata.jl")
include( "../AR-node/autoregression.jl")
include("../AR-node/rules_prototypes.jl")
include("../AR-node/vmp_rules.jl")
include( "../AR-node/observationAR.jl")
include("../helpers/functions.jl")
import Main.ARdata: generateAR
import LinearAlgebra.I, LinearAlgebra.Symmetric

Random.seed!(42)

# define order and generate data
ARorder = 2
diagAR(dim) = Matrix{Float64}(I, dim, dim)
v_x = 1.0 # process noise variance
coefs, data = generateAR(100, ARorder, nvar=v_x, stat=true)
c = zeros(ARorder); c[1] = 1;
# Remove t-1 sample from x
#x_data = [sin(t/10) for t in 1:100]#
x_data = [x[1] for x in data]

v_y = 2 # measurement noise variance
# Observations
y_data = [x + sqrt(v_y)*randn() for x in x_data];

n_samples = length(y_data);
# Building the model
g = FactorGraph()
ForneyLab.draw()

@RV a ~ GaussianMeanPrecision(zeros(ARorder), diagAR(ARorder))
@RV x_0 ~ GaussianMeanPrecision(zeros(ARorder), diagAR(ARorder))

@RV w ~ Gamma(0.0001, 0.0001)

c = zeros(ARorder); c[1] = 1.0
# Observarion model
y = Vector{Variable}(undef, n_samples)
x = Vector{Variable}(undef, n_samples)
z = Vector{Variable}(undef, n_samples)
x_t_min = x_0
for i in 1:n_samples
    global x_t_min
    @RV x[i] = AR(a, x_0, w)
    @RV z[i] = dot(x[i], c)
    @RV y[i] ~ GaussianMeanPrecision(z[i], v_y^-1)
    placeholder(y[i], :y, index=i)
    x_t_min = x[i]
end

# Define recognition factorization
RecognitionFactorization()

q_w = RecognitionFactor(w)
q_a = RecognitionFactor(a)
q_x_0 = RecognitionFactor(x_0)

q_x = Vector{RecognitionFactor}(undef, n_samples)
for t in 1:n_samples
    q_x[t] = RecognitionFactor(x[t])
end

# Generate the variational update algorithms for each recognition factor
# Compile algorithm
algo_w = variationalAlgorithm(q_w, name="W")
algo_a = variationalAlgorithm(q_a, name="A")
algo_x = variationalAlgorithm([q_x_0; q_x], name="X")

eval(Meta.parse(algo_a))
eval(Meta.parse(algo_w))
eval(Meta.parse(algo_x))

data = Dict(:y => y_data)
# Initial recognition distributions
marginals = Dict(:a => ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=zeros(ARorder), w=diagAR(ARorder)),
                 :w => ProbabilityDistribution(Univariate, Gamma, a=0.0001, b=0.0001))

for t in 0:n_samples
    marginals[:x_*t] = vague(GaussianMeanPrecision, ARorder)
end

n_iter = 10
p = Progress(n_iter, 1, "Observed ")
for t in 1:n_iter
    update!(p, t)
    stepX!(data, marginals)
    stepA!(data, marginals)
    stepW!(data, marginals)
end


m_xt = [mean(marginals[:x_*t])[1] for t in 1:n_samples]
v_xt = [cov(marginals[:x_*t])[1] for t in 1:n_samples]

using Plots
from = 1
upto = 100
scatter(y_data, markershape = :xcross, markeralpha = 0.6,
        markersize = 2, xlabel="time t", ylabel="value", label="observations", xlims=(from, upto))
plot!(x_data, color=:magenta, label=L"real \quad x_t", title="AR($ARorder) process")
plot!(m_xt, ribbon=(sqrt.(v_xt), sqrt.(v_xt)),
      linestyle=:dash, linewidth = 2,
      color=:black,
      fillalpha = 0.2,
      fillcolor = :black,
      label="inferred")
