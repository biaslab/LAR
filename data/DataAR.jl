module DataAR

using CSV
using DataFrames
import PolynomialRoots.roots

export loadAR, generateAR, generateHAR, generateSIN, writeAR, readAR

# Filter bad data
FloatParse(x) =
try
    parse(Float64, x)
    return true
catch
    return false
end

function loadAR(filepath::String; col, delim=',')
    df = CSV.File(filepath, delim=delim) |> DataFrame
    x = []
    df = DataFrame(value=df[col])
    # Data
    for i in range(1, stop=size(df, 1))
        if typeof(df[i, 1]) == String && FloatParse(df[i, 1])
            xi = parse(Float64, df[i, 1])
            push!(x, xi)
        elseif typeof(df[i, 1]) == Float64
            xi = convert(Float64, df[i, 1])
            push!(x, xi)
        end
    end
    return x
end

function generate_coefficients(order::Int)
    stable = false
    true_a = []
    # Keep generating coefficients until we come across a set of coefficients
    # that correspond to stable poles
    while !stable
        true_a = randn(order)
        coefs =  append!([1.0], -true_a)
        #reverse!(coefs)
        if false in ([abs(root) for root in roots(coefs)] .> 1)
            continue
        else
            stable = true
        end
    end
    return true_a
end

function generateAR(num::Int, order::Int; nvar=1, stat=true, coefs=nothing)
    if isnothing(coefs) && stat
        coefs = generate_coefficients(order)
    else
        coefs = randn(order)
    end
    inits = randn(order)
    data = Vector{Vector{Float64}}(undef, num+3*order)
    data[1] = inits
    for i in 2:num+3*order
        data[i] = insert!(data[i-1][1:end-1], 1, coefs'data[i-1])
        data[i][1] += sqrt(nvar)*randn()
    end
    data = data[1+3*order:end]
    return coefs, data
end

function generateHAR(num::Int, order::Int; levels=2, nvars=[], stat=true)
    # generate first layer
    if isempty(nvars)
        nvars = ones(levels)
    elseif length(nvars) != levels
        throw(DimensionMismatch("size of variances is not equal to number of levels"))
    end
    θ1, θ2 = generateAR(num, order; nvar=nvars[1], stat=stat)
    data = [θ1, θ2]
    for level in 2:levels
        states = Vector{Vector{Float64}}(undef, num)
        states[1] = randn(order)
        for i in 2:num
            states[i] = insert!(states[i-1][1:end-1], 1, data[level][i]'states[i-1])
            states[i][1] += sqrt(nvars[2])*randn()
        end
        push!(data, states)
    end
    return data
end

function generateSIN(num::Int, noise_variance=1/5)
    coefs = [2cos(1), -1]
    order = length(coefs)
    inits = [sin(1), sin(0)]
    data = Vector{Vector{Float64}}(undef, num+10*order)
    data[1] = inits
    for i in 2:num+10*order
        data[i] = insert!(data[i-1][1:end-1], 1, coefs'data[i-1])
        data[i][1] += sqrt(noise_variance)*randn()
    end
    data = data[1+10*order:end]
    return coefs, data
end

function writeAR(coefs, data; folder=".")
    data = push!(data, coefs)
    df = DataFrame(hcat(data...)')
    order = length(data[1])
    CSV.write(folder*"/AR($order).csv", df)
end

function readAR(filename)
    df = CSV.File(filename) |> DataFrame
    matrix = convert(Matrix, df)
    matrix = [matrix[i, :] for i in 1:size(matrix, 1)]
    coefs = matrix[end]
    data = matrix[1:end-1]
    return coefs, data
end

end  # module
