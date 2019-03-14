mse(x, y) = (sum((x - y).^2))/length(y)

function addNoise(clean; noise_variance)
    noised = [clean[1] .+ sqrt(noise_variance)*randn()]
    for i in 2:length(clean)
        sample = [clean[i][1] + sqrt(noise_variance)*randn()]
        append!(sample, noised[i-1][1:end-1])
        push!(noised, sample)
    end
    return noised
end
