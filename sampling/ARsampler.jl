using Turing
using Plots

function generateAR(num; coef=0.8, nvar=1.0)
    x = Vector(undef, num)
    x[1] = randn()
    for i in 2:num
        x[i] = coef*x[i-1] + sqrt(nvar)*randn()
    end
    return x
end

function generateHAR(num; coef=0.8, nvar=1.0)
    x = Vector(undef, num)
    θ = Vector(undef, num)
    x[1] = randn()
    θ[1] = randn()
    for i in 2:num
        θ[i] = coef*θ[i-1] + sqrt(nvar)*randn()
        x[i] = θ[i]*x[i-1] + sqrt(nvar)*randn()
    end
    return θ, x
end

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


@model HAR(y, onoise=1.0) = begin
    x = Vector(undef, length(observations))
    θ1 = Vector(undef, length(observations))
    v_θ1 ~ InverseGamma(3, 2)
    v_x ~ InverseGamma(3, 2)
    θ2 ~ Normal(0, 1)
    x[1] ~ Normal(0, 1)
    θ1[1] ~ Normal(0, 1)
    for i in 2:length(observations)
        θ1[i] ~ Normal(θ2*θ1[i-1], sqrt(v_θ1))
        x[i] ~ Normal(θ1[i]*x[i-1], sqrt(v_x))
        y[i] ~ Normal(x[i], sqrt(onoise))
    end
    return x, v_θ1, θ1, v_x, θ2
end

# @model HAR(observations) = begin
#     v_θ2 ~ InverseGamma(0.0001, 0.0001)
#     θ_1 ~ Normal(0, 1)
#     v_θ1 ~ InverseGamma(0.0001, 0.0001)
#     θ_1 ~ Normal(0, 1)
#     x = Vector(undef, length(observations))
#     y = Vector(undef, length(observations))
#     x[1] ~ Normal(0, 1)
#     for i in 2:length(observations)
#         x[i] ~ Normal(θ_1*x[i-1], sqrt(v_θ1))
#         y[i] ~ Normal(x[i], 1)
#     end
#     return x, v_θ1, θ_1
# end
using Random

# for i in 1:100000
#     Random.seed!(i)
#     coefs, states = generateHAR(500)
#     if maximum(states) < 1000
#         println(i)
#         break
#     end
# end
Random.seed!(4404)
coefs, states = generateHAR(500)
observations = [st + sqrt(2)*randn() for st in states]
m_har = HAR(observations)

chain = sample(m_har, SMC(), 1000)
#chain1 = sample(m_ar, PG(10), 1000)
#chain2 = sample(m_ar, HMC(0.1, 5), 1000)


m_st = [mean(chain[Symbol("x[$i]")])[2][1] for i in 1:length(states)]
m_θ = [mean(chain[Symbol("θ1[$i]")])[2][1] for i in 1:length(states)]
coef = mean(chain[:θ2])[2][1]
v_x = mean(chain[:v_x])[2][1]
v_θ1 = mean(chain[:v_θ1])[2][1]

scatter(observations, ylims=(minimum(observations), maximum(observations)), xlims=(1, 500))
plot!(states)
plot!(m_st)

plot(coefs, xlims=(1, 200))
plot!(m_θ)

var(m_st)
