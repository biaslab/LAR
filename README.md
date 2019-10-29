Autoregressive (AR) node
============
This repo extends ForneyLab.jl by introducing AR node that can work in three different regimes:
- Sum-Product (Belief propagation) for inferring the coefficients of AR
- Variational Message Passing for inferring AR coefficients and noise variance
- Variational Message Passing (mean-field and structured) for inferring AR coefficients, noise variance and hidden states of AR process.

Getting started
===============
There are [demos](https://github.com/biaslab/VMP-AR/tree/master/demo) available to get you started.

How to connect AR-node to ForneyLab
===============
Super easy: just add these two lines to your code
```julia
include( "../module/autoregressive.jl")
using .AR
```
