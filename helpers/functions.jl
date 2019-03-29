mse(x, y) = sum((x - y).^2)/length(y)

function wmse(x, y, vars=ones(length(y)))
    T = length(y)
    a = [exp(k-T) for k in 1:T]
    a'*((x - y).^2 .* vars)
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
