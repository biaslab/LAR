mse(x, y) = sum((x - y).^2)/length(y)

logPDF(mx, x, vx) = -log(sqrt(2 * pi * vx)) - ((mx - x)^2) / 2 * vx

function wmse(mx, x, vx)
    T = length(x)
    a = [exp(k-T) for k in 1:T]
    a'*((mx - x).^2 ./ vx)
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
