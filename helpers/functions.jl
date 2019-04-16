mse(x, y) = sum((x - y).^2)/length(y)

function wmse(x, y)
    T = length(y)
    a = [exp(k-T) for k in 1:T]
    a'*((x - y).^2)
end

function addNoise(clean; noise_variance)
    noised = [clean[1] .+ sqrt(noise_variance)*randn()]
    for i in 2:length(clean)
        sample = [clean[i][1] + sqrt(noise_variance)*randn()]
        append!(sample, noised[i-1][1:end-1])
        push!(noised, sample)
    end
    return noised
end

function predict(a, x, upto=2)
    predictions = []
    for i in 1:upto
        x̂ = a'*x
        push!(predictions, x̂)
        x = insert!(x[1:end-1], 1, x̂)
    end
    return predictions
end
