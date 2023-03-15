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
import math
import matplotlib.pyplot as plt
import openturns as ot
import openturns.viewer as viewer

assert numpyro.__version__.startswith("0.11.0")
az.style.use("arviz-darkgrid")
sample_size = 100
def Generate_Observational_Data(sample_size):

    alpha = [-4, 4]
    beta = [0.9, 0.7, 1.4, 0.7]  # [Z1, Z2, X, Z2]

    e = dist.Normal(0, 1).sample(jax.random.PRNGKey(0), sample_shape=(sample_size,))
    Z1 = dist.Normal(0, 10).sample(jax.random.PRNGKey(0), sample_shape=(sample_size,))
    Z2 = dist.Normal(0, 15).sample(jax.random.PRNGKey(0), sample_shape=(sample_size,))


    μ_true = beta[0] * Z1 + beta[1] * Z2 + e
    p_true = expit(μ_true)
    X = dist.Bernoulli(p_true).sample(random.PRNGKey(0))
    logit_0 = alpha[0] - beta[2] * X - beta[3]*Z2
    logit_1 = alpha[1] - beta[2] * X - beta[3]*Z2
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = np.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(jax.random.PRNGKey(0), sample_shape=(1,))[0]
    data = pd.DataFrame({"Y": Y, 'X': X, "Z1": Z1, "Z2": Z2})

    return data, X, Z1, Z2, Y

data, X, Z1, Z2, Y = Generate_Observational_Data(sample_size)
print(data)

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
    print(prediction)

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
trace = mcmc.get_samples()

"""Otan paw na iupologisw to marginal byHand prepei na vrw to f(a),f(b0) ktl. Ayto to kanw me auth th synartisi"""

def calculate_LogPDF_of_trace(trace):
    ot.RandomGenerator.SetSeed(1000)
    sample = ot.Sample.BuildFromPoint(trace) #Convert array to openturns object
    ks = ot.KernelSmoothing()
    fittedDist = ks.build(sample)
    f_trace = fittedDist.computeLogPDF(sample)

    return f_trace

# print(sum(calculate_LogPDF_of_trace(numpy.array(trace['cutpoints'][:, 0]))))

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

Log_likelihood = Calculate_Marginal_Likelehood_by_Hand(X,Z1,Z2,Y,trace)
print(Log_likelihood)


def model2(X,Z2,Y,nclasses=3):

    b_X_eta = sample("b_X_eta", Normal(0, 100))
    b_Z2_eta = sample("b_Z2_eta", Normal(0, 100))

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([nclasses - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    prediction = X * b_X_eta + Z2 * b_Z2_eta

    numpyro.sample(
        "Y",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),  # probabbilities that we try to predict
        obs=Y
    )

sampler2 = NUTS(model2)
mcmc2 = MCMC(sampler2, num_samples=sample_size, num_warmup=1000)
mcmc2.run(
    jax.random.PRNGKey(0),
    X=data["X"].to_numpy(),
    Z2=data["Z2"].to_numpy(),
    nclasses=3,
    Y=data['Y'].to_numpy(),
)
mcmc2.print_summary()
trace2 = mcmc2.get_samples()

def Calculate_Marginal_Likelehood_by_Hand2(X,Z2,Y,trace2):
    a_0 = trace2['cutpoints'][:, 0]
    a_1 = trace2['cutpoints'][:, 1]
    logit_0 = a_0 - trace2['b_X_eta'] * X - trace2['b_Z2_eta'] * Z2
    logit_1 = a_1 - trace2['b_X_eta'] * X - trace2['b_Z2_eta'] * Z2
    prob_0 = expit(logit_0)
    prob_1 = expit(logit_1) -prob_0
    prob_2 = 1 - expit(logit_1)

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

Log_likelihood2 = Calculate_Marginal_Likelehood_by_Hand2(X,Z2,Y,trace2)
print(Log_likelihood2)


def model3(X,Z1,Y,nclasses=3):

    b_X_eta = sample("b_X_eta", Normal(0, 100))
    b_Z1_eta = sample("b_Z1_eta", Normal(0, 100))

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([nclasses - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    prediction = X * b_X_eta + Z1 * b_Z1_eta

    numpyro.sample(
        "Y",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),  # probabbilities that we try to predict
        obs=Y
    )

sampler3 = NUTS(model3)
mcmc3 = MCMC(sampler3, num_samples=sample_size, num_warmup=1000)
mcmc3.run(
    jax.random.PRNGKey(0),
    X=data["X"].to_numpy(),
    Z1=data["Z1"].to_numpy(),
    nclasses=3,
    Y=data['Y'].to_numpy(),
)
mcmc3.print_summary()
trace3 = mcmc3.get_samples()

def Calculate_Marginal_Likelehood_by_Hand3(X,Z1,Y,trace3):
    a_0 = trace2['cutpoints'][:, 0]
    a_1 = trace2['cutpoints'][:, 1]
    logit_0 = a_0 - trace3['b_X_eta'] * X - trace3['b_Z1_eta'] * Z1
    logit_1 = a_1 - trace3['b_X_eta'] * X - trace3['b_Z1_eta'] * Z1
    prob_0 = expit(logit_0)
    prob_1 = expit(logit_1) -prob_0
    prob_2 = 1 - expit(logit_1)

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

Log_likelihood3 = Calculate_Marginal_Likelehood_by_Hand3(X,Z1,Y,trace3)
print(Log_likelihood3)