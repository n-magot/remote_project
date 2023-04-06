from scipy.stats import bernoulli, rv_discrete
from scipy.special import expit
import arviz as az
import jax
import jax.numpy as jnp
from jax import random, vmap
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import numpy as np, random
import numpyro
from numpyro import sample, handlers
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS, log_likelihood

numpyro.set_host_device_count(4)
az.style.use("arviz-darkgrid")

sample_size = 800

def Generate_Observational_Data(sample_size):
    #a= [a0 a1] einai to internship, ta cutpoints vasika mporoume na skeftomaste

    alpha = [0, 1]
    beta = [0.9, 0.7, 1.4, 0.5]  # [Z1, Z2, X, Z2]

    Z1 = dist.Normal(0, 10).sample(jax.random.PRNGKey(7), sample_shape=(sample_size,))
    Z2 = dist.Normal(0, 15).sample(jax.random.PRNGKey(7), sample_shape=(sample_size,))
    e = dist.Normal(0, 1).sample(jax.random.PRNGKey(7), sample_shape=(sample_size,))


    μ_true = beta[0] * Z1 + beta[1] * Z2 + e
    p_true = expit(μ_true)
    X = dist.Bernoulli(p_true).sample(random.PRNGKey(4))
    logit_0 = alpha[0] - beta[2] * X - beta[3]*Z2
    logit_1 = alpha[1] - beta[2] * X - beta[3]*Z2
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = np.stack((prob_0, prob_1, prob_2), axis=1)

    data = jnp.array(np.dstack([X, Z1, Z2]))
    labels = dist.Categorical(probs=probs).sample(jax.random.PRNGKey(7), sample_shape=(1,))[0]

    return data, labels

data, labels = Generate_Observational_Data(sample_size)

N, D = sample_size, 3

def oredered_regression(data, labels):

    beta = numpyro.sample('beta', dist.Normal(jnp.zeros(D), jnp.ones(D)))
    intercept = numpyro.sample('intercept', dist.Normal(0., 10.))

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([D - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    logits = jnp.sum(beta * data + intercept, axis=-1)


    return numpyro.sample("obs", dist.OrderedLogistic(predictor=logits, cutpoints=cutpoints), obs=labels)

num_warmup, num_samples = 20, 800
mcmc = MCMC(NUTS(model=oredered_regression), num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(random.PRNGKey(1234), data, labels)
mcmc.print_summary()


def log_likelihood(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['obs']
    return obs_node['fn'].log_prob(obs_node['value'])

def log_predictive_density(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(lambda rng_key, params: log_likelihood(rng_key, params, model, *args, **kwargs))
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return jnp.sum(logsumexp(log_lk_vals, 0) - jnp.log(n))

Log_Likelihood = log_predictive_density(random.PRNGKey(2), mcmc.get_samples(), oredered_regression, data, labels)
print(Log_Likelihood)
