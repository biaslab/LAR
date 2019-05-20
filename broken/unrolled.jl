# FIXME: Offline model doesn't work at the moment

import Main.ARdata: generateAR
using Plots
import LinearAlgebra.I, LinearAlgebra.Symmetric
using Revise
using ForneyLab

# define order and generate data
ARorder = 1
diagAR(dim) = Matrix{Float64}(I, dim, dim)
v_x = 1.0 # process noise variance
coefs, data = generateAR(100, ARorder, nvar=v_x, stat=true)

# Remove t-1 sample from x
x = [x[1] for x in data]

v_y = 2.0 # measurement noise variance
# Observations
y = [x + sqrt(v_y)*randn() for x in x];
n = length(y);
# Building the model
g = FactorGraph()

@RV a ~ GaussianMeanPrecision(0.0*rand(ARorder), (10.0*diagAR(ARorder)))
@RV x_t_prev ~ GaussianMeanPrecision(0.5*rand(ARorder), (10.0*diagAR(ARorder)))
#@RV x_t ~ GaussianMeanPrecision(m_x_t, w_x_t)
#@RV x_t
@RV w ~ Gamma(2, 5)
@RV n_t ~ GaussianMeanPrecision(0.0, 0.000001)

c = zeros(ARorder); c[1] = 1.0
# Observarion model
y = Vector{Variable}(undef, n)
x = Vector{Variable}(undef, n)
for i = 1:n
    global x_t_prev
    @RV x[i] = AR(a, x_t_prev, w)
    @RV y[i] = gain_equality(x[i], ARorder) + n_t
    placeholder(y[i], :y, index=i)
    x_t_prev = x[i]
end

ForneyLab.draw()

q = RecognitionFactorization(a,x,w, ids=[:a, :x, :w])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)
eval(Meta.parse(algo))
display(Meta.parse(algo))
ForneyLab.draw(q.recognition_factors[:a])

cfs = generate_coefficients(UInt16(10))

inits = 5*rand(10)
cfs'inits
