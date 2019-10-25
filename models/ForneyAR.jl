# Unknown process and known measurement noises (UPKM)
# joint estimations of x, θ and γ (process noise)

module ForneyAR

using ProgressMeter
using ForneyLab
using Random
using LinearAlgebra
include( "../ARnode/ARNode.jl")
using .ARNode
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision

export buildGraphAR, inferAR

function buildGraphAR(ARorder)
    graph = FactorGraph()
    # declare priors as random variables
    @RV θ ~ GaussianMeanPrecision(placeholder(:m_θ_t, dims=(ARorder,)),
                                  placeholder(:w_θ_t, dims=(ARorder, ARorder)))

    @RV x_t_prev ~ GaussianMeanPrecision(placeholder(:m_x_t_prev, dims=(ARorder,)),
                                         placeholder(:w_x_t_prev, dims=(ARorder, ARorder)))

    @RV γ ~ Gamma(placeholder(:a_w_t), placeholder(:b_w_t))
    @RV x_t ~ Autoregressive(θ, x_t_prev, γ)
    c = zeros(ARorder); c[1] = 1;
    @RV y_t ~ GaussianMeanPrecision(dot(c, x_t), placeholder(:w_y_t))

    # Placeholder for data
    placeholder(y_t, :y_t)

    # Specify recognition factorization
    q = RecognitionFactorization(θ, x_t, x_t_prev, γ, ids=[:Θ :X_t :X_t_prev :Γ])

    return graph, q
end

function inferAR(r_factorization, observations, obs_noise_var; vmp_iter=5, priors::Dict=Dict(), r_stats=false)

    # Define values for prior statistics
    if length(priors) == 6
        try
            m_θ_0 = priors[:m_θ]; w_θ_0 = priors[:w_θ]
            m_x_prev_0 = priors[:m_x]; w_x_prev_0 = priors[:w_x]
            a_w_0 = priors[:a]; b_w_0 = priors[:b]
        catch
            throw("Dude, your priors' specification is incorrect!")
        end
    else
        println("Dude, I will use uninfomative priors!")
        m_θ_0 = zeros(ARorder)
        w_θ_0 = tiny*diagAR(ARorder)
        m_x_prev_0 = zeros(ARorder)
        w_x_prev_0 = tiny*diagAR(ARorder)
        a_w_0 = 0.00001
        b_w_0 = 0.00001
    end

    m_x_prev = Vector{Vector{Float64}}(undef, length(observations))
    w_x_prev = Vector{Array{Float64, 2}}(undef, length(observations))
    m_θ = Vector{Vector{Float64}}(undef, length(observations))
    w_θ = Vector{Array{Float64, 2}}(undef, length(observations))
    a_w = Vector{Float64}(undef, length(observations))
    b_w = Vector{Float64}(undef, length(observations))

    m_x_t_prev_min = m_x_prev_0
    w_x_t_prev_min = w_x_prev_0
    m_θ_t_min = m_θ_0
    w_θ_t_min = w_θ_0
    a_w_t_min = a_w_0
    b_w_t_min = b_w_0

    marginals = Dict()

    if r_stats
        F = Vector{Float64}(undef, length(observations))
        F_iter = Vector{Array{Float64, 1}}(undef, length(observations))
        algoF = freeEnergyAlgorithm()
        eval(Meta.parse(algoF))
    end

    algo = variationalAlgorithm(r_factorization)
    eval(Meta.parse(algo))

    p = Progress(length(observations), 1, "Observed ")
    for t in 1:length(observations)
        update!(p, t)
        marginals[:θ] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_θ_t_min, w=w_θ_t_min)
        marginals[:x_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_prev_min, w=w_x_t_prev_min)
        marginals[:γ] = ProbabilityDistribution(Univariate, Gamma, a=a_w_t_min, b=b_w_t_min)
        if @isdefined(F); f = Vector{Float64}(undef, vmp_iter) end
        for i = 1:vmp_iter
            data = Dict(:y_t => observations[t],
                        :w_y_t => obs_noise_var^-1,
                        :m_θ_t => m_θ_t_min,
                        :w_θ_t => w_θ_t_min,
                        :m_x_t_prev => m_x_t_prev_min,
                        :w_x_t_prev => w_x_t_prev_min,
                        :a_w_t => a_w_t_min,
                        :b_w_t => b_w_t_min)
            Base.invokelatest(stepX_t!, data, marginals)
            Base.invokelatest(stepΘ!, data, marginals)
            Base.invokelatest(stepΓ!, data, marginals)
            Base.invokelatest(stepX_t_prev!, data, marginals)
            m_θ[t] = unsafeMean(marginals[:θ])
            w_θ[t] = unsafePrecision(marginals[:θ])
            m_x_prev[t] = unsafeMean(marginals[:x_t])
            w_x_prev[t] = unsafePrecision(marginals[:x_t])
            a_w[t] = marginals[:γ].params[:a]
            b_w[t] = marginals[:γ].params[:b]
            m_θ_t_min = m_θ[t]
            w_θ_t_min = w_θ[t]
            m_x_t_prev_min = m_x_prev[t]
            w_x_t_prev_min = w_x_prev[t]
            a_w_t_min = a_w[t]
            b_w_t_min = b_w[t]
            if @isdefined(F); f[i] = Base.invokelatest(freeEnergy, data, marginals) end
        end
        if @isdefined(F)
            F_iter[t] = f
            F[t] = F_iter[t][end]
        end
    end

    if r_stats
        return marginals, F_iter, F,
               Dict(:m_x=>m_x_prev,
                    :w_x=>w_x_prev,
                    :m_θ=>m_θ, :w_θ=>w_θ,
                    :a=>a_w, :b=>b_w)
    else
        return marginals
    end
end

end  # module ARFFG
