module FLAR

include("helpers.jl")
include("DataAR.jl")
include("autoregressive.jl")
using .AR

include("models/ForneyAR.jl")

end  # module FLAR
