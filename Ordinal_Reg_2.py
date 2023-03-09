from scipy.stats import bernoulli, rv_discrete
from scipy.special import expit
import arviz as az
import jax
import jax.numpy as jnp
import numpy
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import numpy as np, random
import numpyro
from numpyro import sample, handlers
from numpyro.distributions import (
    Categorical,
    Dirichlet,
    ImproperUniform,
    Normal,
    OrderedLogistic,
    TransformedDistribution,
    constraints,
    transforms,
)
from numpyro.infer import MCMC, NUTS


numpyro.set_host_device_count(4)
az.style.use("arviz-darkgrid")

def Generate_Observational_Data(sample_size):
    #a= [a0 a1] einai to internship, ta cutpoints vasika mporoume na skeftomaste

    alpha = [-2, 4]
    beta = [0.9, 0.7, 1.4, 0.5]  # [Z1, Z2, X, Z2]

    Z1 = dist.Normal(0, 1).sample(jax.random.PRNGKey(7), sample_shape=(10,))
    Z2 = dist.Normal(0, 1).sample(jax.random.PRNGKey(7), sample_shape=(10,))

    μ_true = beta[0] * Z1 + beta[1] * Z2
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

    Y = dist.Categorical(probs=probs).sample(jax.random.PRNGKey(7), sample_shape=(1,))[0]
    data = pd.DataFrame({"Y": Y, 'X': X, "Z1": Z1, "Z2": Z2})

    return data, X, Z1, Z2, Y

data, X, Z1, Z2, Y = Generate_Observational_Data(20)


def model1(X,Z1,Z2,Y,nclasses=3):

    b_X_eta = sample("b_X_eta", Normal(0, 1))
    b_Z1_eta = sample("b_Z1_eta", Normal(0, 1))
    b_Z2_eta = sample("b_Z2_eta", Normal(0, 1))


    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([nclasses - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    prediction = X * b_X_eta + Z1 * b_Z1_eta + Z2 * b_Z2_eta

    numpyro.sample(
        "Y",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),  # probabbilities that we try to predict
        obs=Y
    )

nclasses=3
mcmc_key = random.PRNGKey(1234)
kernel = NUTS(model1)
mcmc = MCMC(kernel, num_warmup=250, num_samples=10)
mcmc.run(mcmc_key, X, Z1, Z2, Y, nclasses)
mcmc.print_summary()
print(mcmc.get_samples()['b_X_eta'])

sampler = numpyro.infer.NUTS(model1)
mcmc = numpyro.infer.MCMC(sampler, num_samples=10, num_warmup=250)
mcmc.run(
    jax.random.PRNGKey(1234),
    X=data["X"].to_numpy(),
    Z1=data["Z1"].to_numpy(),
    Z2=data["Z2"].to_numpy(),
    nclasses=3,
    Y=data['Y'].to_numpy(),
)
mcmc.print_summary()
print(mcmc.get_samples()['b_X_eta'])