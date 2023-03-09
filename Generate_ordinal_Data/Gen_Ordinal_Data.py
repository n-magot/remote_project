from scipy.stats import bernoulli, rv_discrete
from scipy.special import expit
import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

numpyro.set_host_device_count(4)
az.style.use("arviz-darkgrid")

def Generate_Observational_Data(sample_size):
    #a= [a0 a1] einai to internship, ta cutpoints vasika mporoume na skeftomaste

    alpha = [-2, 4]
    beta = [0.9, 0.7, 1.4, 0.5]  # [Z1, Z2, X,Z2]
    e = np.random.normal(1, 0, sample_size)  # noise

    Z1 = np.random.normal(0, 15, sample_size)
    Z2 = np.random.normal(0, 10, sample_size)
    μ_true = beta[0] * Z1 + beta[1] * Z2
    p_true = expit(μ_true)
    X = bernoulli.rvs(p_true)
    logit_0 = alpha[0] - beta[2] * X - beta[3]*Z2
    logit_1 = alpha[1] - beta[2] * X - beta[3]*Z2
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = np.stack((prob_0, prob_1, prob_2), axis=1)
    # Y = []
    # for i in range(sample_size):
    #     y = rv_discrete(values=([1, 2,  3], probs[i])).rvs() #Categorical distribution
    #     Y.append(y)
    # Y = np.array(Y)
    Y = dist.Categorical(probs=probs).sample(jax.random.PRNGKey(7), sample_shape=(1,))[0]
    data = pd.DataFrame({"Y": Y, 'X': X, "Z1": Z1, "Z2": Z2})
    print(data)

    return data

data = Generate_Observational_Data(20)
