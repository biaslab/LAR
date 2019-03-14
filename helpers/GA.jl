
module GA
using Distributions

    mutable struct Chromosome
        genes :: Vector{Float64}
    end

    function mutation(chr :: Chromosome)
        mchr = deepcopy(chr)
        idx = rand(1:length(mchr.genes))
        mchr.genes[idx] = 0.0
        d = Distributions.Uniform(1.0e-12, 1.0e2)
        while mchr.genes[idx] == 0.0
            mchr.genes[idx] = rand(d)
        end
        return mchr
    end


    # Strong type crossover
    function crossover(chr1 :: T, chr2 :: T, n_successors = 4) where T<:Chromosome
        rng = 1:length(chr1.genes)
        children = []
        for i in 1:div(n_successors, 2)
            idxs = collect(Set(rand(rng, rand(rng))))
            child1 = deepcopy(chr1)
            child2 = deepcopy(chr2)
            for idx in idxs
                child1.genes[idx] = deepcopy(chr2.genes[idx])
                child2.genes[idx] = deepcopy(chr1.genes[idx])
            end
            push!(children, child1)
            push!(children, child2)
        end
        return children
    end

    function evolution(population, fitfunction::Function, generations=1000, mutation_p=0.01)
        generation = 1
        pop_sise = length(population)
        fitnesses = Vector{Float64}(undef, length(population))
        best_ind = Nothing
        fitness = 1e10
        while fitness > 0.001 && generation < generations

            # select best 20%
            best = sort(population, by=i->(fitnesses))[1:Int64(floor(0.2*length(population)))]
            # crossover
            population = deepcopy(best)
            birth_card = collect(1:length(best))
            while length(population) != pop_sise
                # select 2 parents
                idx1 = rand(1:length(best))
                deleteat!(birth_card, birth_card .== idx1)
                idx2 = rand(1:length(best))
                deleteat!(birth_card, birth_card .== idx2)
                children = crossover(population[idx1], population[idx2])
                for child in children
                    push!(population, child)
                end
            end

            for individual in population
                if rand(0:0.000001:1) < mutation_p
                    display("mutatated")
                    mutation(individual)
                end
            end

            generation +=1
            if mod(generation, 100) == 0
                mutation_p *= 9.8
            end
            # evaluate fitness
            for (i, ind) in enumerate(population)
                fitnesses[i] = fitfunction(ind)
            end

            best_ind = findmin(fitnesses)
            fitness = best_ind[1]
            println("Best fintess = ", fitness)
            println("Gen #", generation)
        end
        return best_ind, population
    end

    function generate(restrictions, dim)
        d = Distributions.Uniform(1.0e-12, 1.0)
        chr = Chromosome(rand(d, dim))
        for (i, gene) in enumerate(chr.genes)
            if i in restrictions
                chr.genes[i] = abs(chr.genes[i])
            end
        end
        return chr
    end
end  # module GA
