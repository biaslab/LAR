# FIXME: Offline model doesn't work at the moment

include("../data/ARdata.jl")
include( "../AR-node/autoregression.jl")
include("../AR-node/rules_prototypes.jl")
include("../AR-node/vmp_rules.jl")
include( "../AR-node/observationAR.jl")
include("../helpers/functions.jl")
import Main.ARdata: generateAR
using Plots
import LinearAlgebra.I, LinearAlgebra.Symmetric
using Revise
using ForneyLab

# define order and generate data
ARorder = 2
diagAR(dim) = Matrix{Float64}(I, dim, dim)
v_x = 1.0 # process noise variance
coefs, data = generateAR(100, ARorder, nvar=v_x, stat=true)
c = zeros(ARorder); c[1] = 1;
# Remove t-1 sample from x
x = [x[1] for x in data]

v_y = 2.0 # measurement noise variance
# Observations
y_data = [x + sqrt(v_y)*randn() for x in x];
n = length(y);
# Building the model
g = FactorGraph()
ForneyLab.draw()

@RV a ~ GaussianMeanPrecision(zeros(ARorder), diagAR(ARorder))
@RV x_t_prev ~ GaussianMeanPrecision(zeros(ARorder), diagAR(ARorder))
@RV w ~ Gamma(2, 5)

c = zeros(ARorder); c[1] = 1.0
# Observarion model
y = Vector{Variable}(undef, n)
x = Vector{Variable}(undef, n)
z = Vector{Variable}(undef, n)
for i = 1:2
    global x_t_prev
    @RV x[i] = AR(a, x_t_prev, w)
    @RV z[i] = dot(x[i], c)
    @RV y[i] ~ GaussianMeanPrecision(z[i], v_y^-1)
    placeholder(y[i], :y, index=i)
    x_t_prev = x[i]
end

ForneyLab.draw()

q = RecognitionFactorization(a, x_t_prev, w, ids=[:A, :X, :W])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)
eval(Meta.parse(algo))
#display(Meta.parse(algo))
#ForneyLab.draw(q.recognition_factors[:a])

data = Dict(:y => y_data)

# Initial recognition distributions
marginals = Dict(:a => vague(GaussianMeanPrecision, ARorder),
                 :w => vague(Gamma),
                 :x_t_prev => vague(GaussianMeanPrecision, ARorder))

for i in 1:2
    display(marginals)
    display(data)
    stepX!(data, marginals)
    stepA!(data, marginals)
    stepW!(data, marginals)
end
