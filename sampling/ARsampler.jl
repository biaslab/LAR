using Turing
using Plots
using Random
using LinearAlgebra
using LaTeXStrings
include("../data/DataAR.jl")
import Main.DataAR: loadAR, generateAR, generateHAR, generateSIN
diagAR(dim) = Matrix{Float64}(I, dim, dim)

@model AR(observations) = begin
    v_θ1 ~ InverseGamma(3, 2)
    θ_1 ~ Normal(0, 1)
    x = Vector(undef, length(observations))
    y = Vector(undef, length(observations))
    x[1] ~ Normal(0, 1)
    for i in 2:length(observations)
        x[i] ~ Normal(θ_1*x[i-1], sqrt(v_θ1))
        observations[i] ~ Normal(x[i], 1)
    end
    return x, v_θ1, θ_1
end


@model HAR(y, order=1, onoise=2.0) = begin
    x = Vector(undef, length(y))
    θ1 = Vector(undef, length(y))
    v_θ1 ~ InverseGamma(3, 2)
    v_x ~ InverseGamma(3, 2)
    θ2 ~ MvNormal(zeros(order), diagAR(order))
    for i in 1:order
        x[i] ~ Normal(0, 1)
        θ1[i] ~ Normal(1, 1)
    end
    for i in order+1:length(y)
        θ1[i] ~ Normal(θ2'reverse(θ1[i-order:i-1]), sqrt(v_θ1))
        x[i] ~ Normal(reverse(θ1[i-order+1:i])'reverse(θ1[i-order:i-1]), sqrt(v_x))
        y[i] ~ Normal(x[i], sqrt(onoise))
    end
    return x, v_θ1, θ1, v_x, θ2
end

Random.seed!(42)

HARorder = 1
ARorder = HARorder

v_θ1 = 0.5
v_x = 1.0

dataHAR = generateHAR(1000, HARorder, levels=2, nvars=[v_θ1, v_x], stat=true)
coefs = dataHAR[1]
θ = [θ[1] for θ in dataHAR[2]]
x = [x[1] for x in dataHAR[3]]
# Observations
v_y = 2.0
y = [xi[1] + sqrt(v_y)*randn() for xi in dataHAR[3]];

m_har = HAR(y, 1, 2.0)

chain = sample(m_har, SMC(), resampler_threshold=.3, 1000)
m_st = [mean(chain[Symbol("x[$i]")].value) for i in 1:length(x)]
v_st = [std(chain[Symbol("x[$i]")].value) for i in 1:length(x)]
m_θ = [mean(chain[Symbol("θ1[$i]")].value) for i in 1:length(θ)]
v_θ = [std(chain[Symbol("θ1[$i]")].value) for i in 1:length(θ)]
coef = mean(chain[:θ2])[2]
#v_x = mean(chain[:v_x])[2][1]
#v_θ1 = mean(chain[:v_θ1])[2]

logPDF(mx, x, vx) = -0.5*log(2*pi*vx) - ((mx - x)^2)/(2 * vx)
logPDFsHAR = [logPDF(x[t], m_st[t], v_st[t]) for t in 1:length(y)]
sum(logPDFsHAR)
from = 630
upto = 1000
scatter(y, ylims=(minimum(y[from:upto]), maximum(y[from:upto])), markershape = :xcross, markeralpha = 0.6,
        markersize = 2, xlabel="time t", ylabel="value", label="observations", xlims=(from, upto))
plot!(x, color=:magenta, label=L"real \quad x_t", title="AR($ARorder) process")
plot!(m_st, ribbon=sqrt.(v_st),
      linestyle=:dash, linewidth = 2,
      color=:black,
      fillalpha = 0.2,
      fillcolor = :black,
      label="inferred")



plot(xlims=(from, upto), )
plot!(θ[2:end], color=:red, label=L"true \quad \theta^{(1)}", title="Generated AR($ARorder) process")
plot!(m_θ, ribbon=sqrt.(v_θ),
      linestyle=:dash, linewidth = 2,
      color=:black,
      fillalpha = 0.2,
      fillcolor = :black,
      label="inferred")
