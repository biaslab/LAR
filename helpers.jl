using Plots

function plotter(ma, x, mw, testPoints=100, plt=true)
    from = length(x) - testPoints
    predicted = [ForneyLab.sample(marginals[:a])'x + var(marginals[:w]) for x in x[from-1:end-1]]
    actual = [x[1] for x in x[from:end]]
    mse = (sum((predicted - actual).^2))/length(predicted)
    if plt
        ylims!(-1, 2)
        plot!([actual, predicted], title = "unforeseen data", xlabel="days", ylabel="temperature", label=["actual", "predicted"])
    end
    return mse
end

mse(x, y) = (sum((x - y).^2))/length(y)
