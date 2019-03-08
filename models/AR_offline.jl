# FIXME: Offline model doesn't work at the moment

import ARdata: generate_data, generate_coefficients
using Plots
import LinearAlgebra.I, LinearAlgebra.Symmetric
using Revise
using ForneyLab

# define order and generate data
ARorder = UInt16(10)
diagAR(dim) = Matrix{Float64}(I, dim, dim)
coefs, x = generate_data(UInt64(100), ARorder, 100)

# splitting the data
# FIXME
x_series = x[1]
for sample in x
    append!(x_series, sample[end])
end
x_series = x_series[div(length(x_series), 2):end]

#plot(collect(1:length(x_series)), x_series)

# actual data
#x = x[div(length(x), 2):end]

# Observations
#y = [xi[end] for xi in x[2:end]] .+ rand()*0.000001
n = length(x); n = 2
# Building the model
g = FactorGraph()

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

# Observarion model
x = Vector{Variable}(undef, n)
for t = 1:n
    global x_t_prev
    @RV x[t]
    Autoregression(x[t], x_t_prev, a, w)
    placeholder(x[t], :y, index=t)
    x_t_prev = x[t]
end

placeholder(m_x_t_prev, :m_x_t_prev, dims=(ARorder,))
placeholder(w_x_t_prev, :w_x_t_prev, dims=(ARorder, ARorder))
placeholder(a_w_t, :a_w_t)
placeholder(b_w_t, :b_w_t)
placeholder(m_a_t, :m_a_t, dims=(ARorder,))
placeholder(w_a_t, :w_a_t, dims=(ARorder, ARorder))

ForneyLab.draw()

q = RecognitionFactorization(a, x, w, ids=[:a, :x, :w])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)
ForneyLab.draw(q.recognition_factors[:a])
eval(Meta.parse(algo))
display(Meta.parse(algo))
ForneyLab.draw(q.recognition_factors[:a])

cfs = generate_coefficients(UInt16(10))

inits = 5*rand(10)
cfs'inits
