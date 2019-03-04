module ARdata

using CSV, DataFrames, PolynomialRoots

# Filter bad data
FloatParse(x) = try
    parse(Float64, x)
    return true
catch
    return false
end

function use_data(filepath::String, order::UInt16)

    df = CSV.File(filepath) |> DataFrame
    x = []

    names!(df, [:timestamp, :value])

    filter!(row -> FloatParse(row[:value]) == true, df)

    # Data
    for i in range(1, size(df, 1) - order)
        xi = map(x->parse(Float64,x), df[i:order+i - 1, :value])
        push!(x, xi)
    end
    return x
end

function generate_coefficients(order::UInt16)
    stable = false
    true_a = []
    # Keep generating coefficients until we come across a set of coefficients
    # that correspond to stable poles
    while !stable
        true_a = rand(order) .- .5
        coefs =  append!([1.0], -true_a)
        reverse!(coefs)
        if maximum([abs(root) for root in roots(coefs)]) < 1
            stable = true
        end
    end
    return reverse(true_a)
end


function generate_data(num::UInt64, order::UInt16, scale, noise_variance=1)
    coefs = generate_coefficients(order)
    #coefs = [-0.10958935, -0.34564819,  0.3682048,   0.3134046,  -0.21553732,  0.34613629, 0.41916508,  0.0165352,   0.14163503, -0.38844378]
    inits = scale*rand(order)
    data = Vector{Vector{Float64}}(undef, num+100*order)
    data[1] = inits
    for i in 2:num+100*order
        data[i] = insert!(data[i-1][1:end-1], 1, coefs'data[i-1])
        data[i][1] += noise_variance * rand()
    end
    data = data[1+100*order:end]
    return coefs, data
end

end  # module
