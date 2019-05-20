# Unknown process and known measurement noises (UPKM)
# joint estimations of x, a and w (process noise)

using ProgressMeter
using ForneyLab
using Random
using LinearAlgebra
include( "../AR-node/autoregression.jl")
include("../AR-node/rules_prototypes.jl")
include("../AR-node/vmp_rules.jl")
include( "../AR-node/observationAR.jl")
include("../helpers/functions.jl")
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision

function buildGraphAR(ARorder)
    graph = FactorGraph()
    # declare priors as random variables
    @RV m_x_t_prev
    @RV w_x_t_prev
    @RV m_a_t
    @RV w_a_t
    @RV m_y_t
    @RV w_y_t
    @RV a_w_t
    @RV b_w_t
    @RV a ~ GaussianMeanPrecision(m_a_t, w_a_t)
    @RV x_t_prev ~ GaussianMeanPrecision(m_x_t_prev, w_x_t_prev)
    @RV w ~ Gamma(a_w_t, b_w_t)
    @RV x_t = AR(a, x_t_prev, w)
    observationAR(m_y_t, x_t, w_y_t)

    # Placeholders for prior
    placeholder(m_a_t, :m_a_t, dims=(ARorder,))
    placeholder(w_a_t, :w_a_t, dims=(ARorder, ARorder))
    # Placeholder for data
    placeholder(m_x_t_prev, :m_x_t_prev, dims=(ARorder,))
    placeholder(w_x_t_prev, :w_x_t_prev, dims=(ARorder, ARorder))
    placeholder(a_w_t, :a_w_t)
    placeholder(b_w_t, :b_w_t)
    placeholder(m_y_t, :m_y_t)
    placeholder(w_y_t, :w_y_t)

    # Specify recognition factorization
    q = RecognitionFactorization(a, x_t, x_t_prev, w, ids=[:A :X_t :X_t_prev :W])

    return graph, q
end

function inferAR(r_factorization, observations, obs_noise_var; vmp_iter=5, priors::Dict=Dict(), r_stats=false)

    # Define values for prior statistics
    if length(priors) == 6
        try
            m_a_0 = priors[:m_a]; w_a_0 = priors[:w_a]
            m_x_prev_0 = priors[:m_x]; w_x_prev_0 = priors[:w_x]
            a_w_0 = priors[:a]; b_w_0 = priors[:b]
        catch
            throw("Incorrect input for priors")
        end
    else
        println("Using uninformative priors")
        m_a_0 = zeros(ARorder)
        w_a_0 = tiny*diagAR(ARorder)
        m_x_prev_0 = zeros(ARorder)
        w_x_prev_0 = tiny*diagAR(ARorder)
        a_w_0 = 0.00001
        b_w_0 = 0.00001
    end

    m_x_prev = Vector{Vector{Float64}}(undef, length(observations))
    w_x_prev = Vector{Array{Float64, 2}}(undef, length(observations))
    m_a = Vector{Vector{Float64}}(undef, length(observations))
    w_a = Vector{Array{Float64, 2}}(undef, length(observations))
    a_w = Vector{Float64}(undef, length(observations))
    b_w = Vector{Float64}(undef, length(observations))

    m_x_t_prev_min = m_x_prev_0
    w_x_t_prev_min = w_x_prev_0
    m_a_t_min = m_a_0
    w_a_t_min = w_a_0
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

    p = Progress(length(y), 1, "Observed ")
    for t in 1:length(observations)
        update!(p, t)
        marginals[:a] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_a_t_min, w=w_a_t_min)
        marginals[:x_t_prev] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m_x_t_prev_min, w=w_x_t_prev_min)
        marginals[:w] = ProbabilityDistribution(Univariate, Gamma, a=a_w_t_min, b=b_w_t_min)
        if @isdefined(F); f = Vector{Float64}(undef, vmp_iter) end
        for i = 1:vmp_iter
            data = Dict(:m_y_t => observations[t],
                        :w_y_t => obs_noise_var^-1,
                        :m_a_t => m_a_t_min,
                        :w_a_t => w_a_t_min,
                        :m_x_t_prev => m_x_t_prev_min,
                        :w_x_t_prev => w_x_t_prev_min,
                        :a_w_t => a_w_t_min,
                        :b_w_t => b_w_t_min)
            stepX_t!(data, marginals)
            stepA!(data, marginals)
            stepW!(data, marginals)
            stepX_t_prev!(data, marginals)
            m_a[t] = unsafeMean(marginals[:a])
            w_a[t] = unsafePrecision(marginals[:a])
            m_x_prev[t] = unsafeMean(marginals[:x_t])
            w_x_prev[t] = unsafePrecision(marginals[:x_t])
            a_w[t] = marginals[:w].params[:a]
            b_w[t] = marginals[:w].params[:b]
            m_a_t_min = m_a[t]
            w_a_t_min = w_a[t]
            m_x_t_prev_min = m_x_prev[t]
            w_x_t_prev_min = w_x_prev[t]
            a_w_t_min = a_w[t]
            b_w_t_min = b_w[t]
            if @isdefined(F); f[i] = freeEnergy(data, marginals) end
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
                    :m_a=>m_a, :w_a=>w_a,
                    :a=>a_w, :b=>b_w)
    else
        return marginals
    end
end
