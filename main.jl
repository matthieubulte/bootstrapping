using Distributions
using Plots
using LinearAlgebra

function sampleData(β, σ², nsamples)
    x = rand(Normal(), nsamples);
    ϵ = rand(Normal(0, σ²), nsamples);
    y = β*x + ϵ;
    
    x, y, ϵ
end

function sampleDataSigmaPrior(β, nsamples)
    x = rand(Normal(), nsamples);
    σ²ᵢ = rand(Chisq(1), nsamples)
    ϵ = rand(Normal(0, σ²ᵢ), nsamples);
    y = β*x + ϵ;
    
    x, y, ϵ
end

function sampleDataErrorCorrelatedToCovariates(β, γ, nsamples)
    x = rand(Normal(), nsamples);
    σ²ᵢ = γ*x + rand(Normal(0,1), nsamples)
    ϵ = rand(Normal(0, σ²ᵢ), nsamples);
    y = β*x + ϵ;
    
    x, y, ϵ
end

function sampleDataTwoVariances(β, σ²₁, σ²₂, nsamples)
    x = rand(Normal(), nsamples);

    σ²_idx = 1 .+ rand(Bernoulli(), nsamples) * 1
    σ²ᵢ = [σ²₁; σ²₂][σ²_idx]

    ϵ = sqrt.(σ²ᵢ) .* rand(Normal(0.0, 1.0), nsamples);
    y = β*x + ϵ;
    
    x, y, ϵ, σ²ᵢ
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

# The OLS estimator of β follows sqrt(n) * (β - β̂) ~ N(0, σ²)
β̂(data)

# this estimates the variance of β̂, which should be equal to σ²/n = 1/n
true_σ̂² = 1/dot(x, x)

# this is a bootstrap estimator of the variance of  β̂
function bootstrap_σ̂²(data)
    samples = bootstrap(β̂, data, nbootstrap, bootstrapsamplesize);
    var(samples)
end

bootstrap_σ̂²(data)

# we bootstrap the distribution of the bootstrap estimator of the variance of β̂
sample_σ̂² = bootstrap(_ -> bootstrap_σ̂²(data), data, nbootstrap, 1);
mean(sample_σ̂²), var(sample_σ̂²)


# heteroskedasticity
# the true variance of each observation is either 1.0 or 2.0
x, y, ϵ, Ω = sampleDataTwoVariances(β, 1.0, 2.0, datasamplesize);
data = hcat(x, y);

β̂(data)

# so now, the OLS estimator of the variance of β̂ is wrong...
1/dot(x, x)

# ...which we can see when constructing the bootstrap estimate of it
bootstrap_σ̂²(data)


# beta_ols
#     unbiased estimator of beta
#     estimator of variance of beta_ols is s^2 / x'x
#
#     homoskedasticity
#         ols estimator of variance of the beta_ols is consistent
    
#     heteroskedasticity
#         ols estimator of variance of the beta_ols is  not consistent

# beta_gls
#     unbiased estimator of beta
#     estimator of variance of the beta_gls is 




# but if we know the real σ²ᵢ we can compute the gls
1/dot(x, x./Ω)


