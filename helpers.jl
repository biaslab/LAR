mse(x, y) = (sum((x - y).^2))/length(y)

function addNoise(clean; noise_variance)
    noised = [clean[1] .+ sqrt(0.00001)*randn()]
    for i in 2:length(clean)
        sample = [clean[i][1] + sqrt(0.00001)*randn()]
        append!(sample, noised[i-1][1:end-1])
        push!(noised, sample)
    end
    return noised
end
