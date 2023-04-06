"""Ayto einai pou kati den moy paei kala kai prin kanw tis allages 15/03"""
from scipy.special import expit
import arviz as az
import jax
import numpy
import numpyro.distributions as dist
import pandas as pd
from jax import numpy as np, random
import numpyro
from numpyro import sample
from numpyro.distributions import (Normal)
from numpyro.infer import MCMC, NUTS
from numpyro import sample, handlers
from jax.scipy.special import logsumexp
from itertools import combinations
# import jax.numpy as jnp
from jax import random, vmap
import math

assert numpyro.__version__.startswith("0.11.0")
az.style.use("arviz-darkgrid")
sample_size = 300
rng_key = jax.random.PRNGKey(0)



def Generate_Observational_Data(sample_size):

    alpha = [-4, 4]
    beta = [0.9, 0.7, 1.4, 0.7]  # [Z1, Z2, X, Z2]

    e = dist.Normal(0, 1).sample(rng_key, sample_shape=(sample_size,))
    Z1 = dist.Normal(0, 10).sample(rng_key, sample_shape=(sample_size,))
    Z2 = dist.Normal(0, 15).sample(rng_key, sample_shape=(sample_size,))


    μ_true = beta[0] * Z1 + beta[1] * Z2 + e
    p_true = expit(μ_true)
    X = dist.Bernoulli(p_true).sample(rng_key)
    logit_0 = alpha[0] - beta[2] * X - beta[3]*Z2
    logit_1 = alpha[1] - beta[2] * X - beta[3]*Z2
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = np.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(rng_key, sample_shape=(1,))[0]
    data = pd.DataFrame({"Y": Y, 'X': X, "Z1": Z1, "Z2": Z2})
    data_X = np.stack((X, Z1, Z2), axis=1)
    data_Y = dist.Categorical(probs=probs).sample(rng_key, sample_shape=(1,))[0]

    return data, data_X, data_Y, Y, X, Z1, Z2

"""!!!  Oti dataframe kai na diavazoume tha vazoume stin prwti stili
to outcome(Y) kai sthn 2h to treatment(X) !!!"""
data, data_X, data_Y, Y, X, Z1, Z2 = Generate_Observational_Data(sample_size)

"""Take all possible combinations for regression """
#Pairnei to dataset kai ftiaxnei olous tous pithanous syndiasmous pou periexoun to X
# wste na kanw ta regression me 1 function

sample_list = data.columns.values[1:]
list_comb = []
for i in range(data.shape[1]-1):
    list_combinations = list(combinations(sample_list, data.shape[1]-i-1))
    for x in list_combinations:
        if x[0] == 'X':
            list_comb.append(x)
print('The combinations of regression models for Y are {}'.format(list_comb))

num_warmup, num_samples = 1000, sample_size


def Regression_cases(Y, X, Z1, Z2, reg_variables):

    vars_dict = {'Y': Y, 'X': X, 'Z1': Z1, 'Z2': Z2}
    beta = []
    for i in range(len(reg_variables)):
        beta.append(sample('beta_{}'.format(reg_variables[i]), Normal(0, 100)))

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([N_classes - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    prediction = 0
    for i in range(len(reg_variables)):
        prediction = prediction + beta[i] * vars_dict['{}'.format(reg_variables[i])]

    numpyro.sample(
        "Y",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),  # probabbilities that we try to predict
        obs=Y
    )


def log_likelihood(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['Y']
    return sum(obs_node['fn'].log_prob(obs_node['value']))


"""Regression for (X,Z1,Z2)"""
for i in range(len(list_comb)):
    reg_variables = list_comb[i]

    N_coef, N_classes = len(reg_variables), 3

    mcmc = MCMC(NUTS(model=Regression_cases), num_warmup=num_warmup, num_samples=num_samples)
    #Edw orizw poia thelw na einai ta data_X mou gia to kathe regression
    mcmc.run(rng_key, Y, X, Z1, Z2, reg_variables)
    mcmc.print_summary()
    trace = mcmc.get_samples()

    Log_Likelihood = log_likelihood(rng_key, trace, Regression_cases, Y, X, Z1, Z2, reg_variables)
    print("Log Likelihood for {}:".format(reg_variables), Log_Likelihood)

def model1(X,Z1,Z2,Y,nclasses=3):

    b_X_eta = sample("b_X_eta", Normal(0, 100))
    b_Z1_eta = sample("b_Z1_eta", Normal(0, 100))
    b_Z2_eta = sample("b_Z2_eta", Normal(0, 100))


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

sampler = NUTS(model1)
mcmc = MCMC(sampler, num_samples=sample_size, num_warmup=1000)
mcmc.run(
    jax.random.PRNGKey(0),
    X=data["X"].to_numpy(),
    Z1=data["Z1"].to_numpy(),
    Z2=data["Z2"].to_numpy(),
    nclasses=3,
    Y=data['Y'].to_numpy(),
)
mcmc.print_summary()
trace1 = mcmc.get_samples()

def Calculate_Marginal_Likelehood_by_Hand(X,Z1,Z2,Y,trace):
    a_0 = trace['cutpoints'][:, 0]
    a_1 = trace['cutpoints'][:, 1]
    logit_0 = a_0 - trace['b_X_eta'] * X - trace['b_Z1_eta'] * Z1 - trace['b_Z2_eta'] * Z2
    logit_1 = a_1 - trace['b_X_eta'] * X - trace['b_Z1_eta'] * Z1 - trace['b_Z2_eta'] * Z2
    prob_0 = expit(logit_0)
    prob_1 = expit(logit_1) -prob_0
    prob_2 = 1 - expit(logit_1)

    #Calculate Likelihood
    P = 1
    for i in range(len(prob_0)):
        if Y[i] == 0:
            P = P * prob_0[i]
        elif Y[i] == 1:
            P = P * prob_1[i]
        else:
            P = P * prob_2[i]

    Log_Likelihood = math.log(P)

    return Log_Likelihood

Log_likelihood = Calculate_Marginal_Likelehood_by_Hand(X,Z1,Z2,Y,trace1)
print(Log_likelihood)
