module ARdata
import PolynomialRoots.roots
import CSV: File
import DataFrames: DataFrame, names!, dropmissing!

# Filter bad data
FloatParse(x) =
try
    parse(Float64, x)
    return true
catch
    return false
end

function load(filepath::String, order::Int; col, delim=',')
    df = File(filepath, delim=delim) |> DataFrame
    x = []
    df = DataFrame(value=df[col])
    #dropmissing!(df)
    if typeof(df[1, 1]) == String
        filter!(row -> FloatParse(row[:value]) == true, df)
    end
    # Data
    for i in range(1, stop=size(df, 1) - order)
        if typeof(df[i, 1]) == String
            xi = map(x->parse(Float64,x), df[i:order+i - 1, :value])
        elseif typeof(df[i, 1]) == Float64
            xi = map(x->convert(Float64,x), df[i:order+i - 1, :value])
        end
        push!(x, xi)
    end
    return x
end

function generate_coefficients(order::Int)
    stable = false
    true_a = []
    # Keep generating coefficients until we come across a set of coefficients
    # that correspond to stable poles
    while !stable
        true_a = randn(order) .- .5
        coefs =  append!([1.0], -true_a)
        reverse!(coefs)
        if false in ([abs(root) for root in roots(coefs)] .< 1)
            continue
        else
            stable = true
        end
    end
    return true_a
end

function generate(num::Int, order::Int, scale::Real; noise_variance=1)
    coefs = generate_coefficients(order)
    inits = scale*randn(order)
    data = Vector{Vector{Float64}}(undef, num+3*order)
    data[1] = inits
    for i in 2:num+3*order
        data[i] = insert!(data[i-1][1:end-1], 1, coefs'data[i-1])
        data[i][1] += sqrt(noise_variance)*randn()
    end
    data = data[1+3*order:end]
    return coefs, data
end

function generate_sin(num::Int, noise_variance=1/5)
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

function dump(data, coefs; folder=".")
    data = push!(data, coefs)
    df = DataFrame(hcat(data...)')
    order = length(data[1])
    CSV.write("AR($order).csv", df)
end

function read_dump(filename)
    df = File("AR(2).csv") |> DataFrame
    matrix = convert(Matrix, df)
    matrix = [matrix[i, :] for i in 1:size(matrix, 1)]
    coefs = matrix[end]
    data = matrix[1:end]
    return coefs, data
end

end  # module
