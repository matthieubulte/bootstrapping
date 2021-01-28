using Distributions
using Plots
using LinearAlgebra

function sampleData(β, σ², nsamples; hslope=0.0)
    x = rand(Normal(), nsamples);
    hetero = 1.0 .+ hslope .* x
    ϵ = hetero .* rand(Normal(0, σ²), nsamples);
    y = β*x + ϵ;
    
    x, y, ϵ
end

function bootstrap(f, data, nbootstrap, bootstrapsamplesize)
    samples = zeros(nbootstrap)
    nsamples = size(data)[1]
    for i = 1:nbootstrap
        idx = rand(DiscreteUniform(1, nsamples), bootstrapsamplesize)
        data_i = data[idx, :]
        samples[i] = f(data_i)
    end
    samples
end

datasamplesize = 1000;
nbootstrap = 1000;
bootstrapsamplesize = 800;

β = 3
σ² = 1
β̂ = data -> dot(data[:, 1], data[:, 2]) / dot(data[:, 1], data[:, 1])


# homoskedasticity
x, y, ϵ = sampleData(β, σ², datasamplesize);
data = hcat(x, y);

β̂(data)
true_σ̂² = 1/dot(x, x)

function bootstrap_σ̂²(data)
    samples = bootstrap(β̂, data, nbootstrap, bootstrapsamplesize);
    var(samples)
end

bootstrap_σ̂²(data)

# we bootstrap the distribution of the bootstrap estimator
sample_σ̂² = bootstrap(_ -> bootstrap_σ̂²(data), data, nbootstrap, 1);
mean(sample_σ̂²), var(sample_σ̂²)


# heteroskedasticity
x, y, ϵ = sampleData(β, σ², nsamples; hslope=1.0);
data = hcat(x, y);

β̂(data)
1/dot(x, x)

bootstrap_σ̂²(data)
