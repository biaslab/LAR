module ARdata
import PolynomialRoots.roots
import CSV: File
import DataFrames: DataFrame, names!

# Filter bad data
FloatParse(x) = try
    parse(Float64, x)
    return true
catch
    return false
end

function use_data(filepath::String, order::Int)
    df = File(filepath) |> DataFrame
    x = []
    names!(df, [:timestamp, :value])
    filter!(row -> FloatParse(row[:value]) == true, df)
    # Data
    for i in range(1, stop=size(df, 1) - order)
        xi = map(x->parse(Float64,x), df[i:order+i - 1, :value])
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
    display(true_a)
    return true_a
end


function generate_data(num::Int, order::Int, scale::Real; noise_variance=1)
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

function stacked_data(num::Int, order::Int, scale::Real; coefs, noise_variance=1)
    inits = scale*randn(order)
    data = Vector{Vector{Float64}}(undef, num)
    data[1] = insert!(inits[1:end-1], 1, coefs[1]'inits)
    for i in 2:num
        display(i)
        data[i] = insert!(data[i-1][1:end-1], 1, coefs[i]'data[i-1])
        data[i][1] += sqrt(noise_variance)*randn()
    end
    return data
end

end  # module
