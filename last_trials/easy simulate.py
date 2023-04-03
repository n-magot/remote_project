"""DImioyrgia polu aplou dataset me mia continuous independent variable X kai to Y na pairnei times analoga me to X"""
import numpy as np
from scipy.special import expit
import arviz as az
import jax
import numpyro.distributions as dist
import pandas as pd
from jax import numpy as np, random
import numpyro
from numpyro.distributions import (Normal)
from numpyro.infer import MCMC, NUTS
from numpyro import sample, handlers
import math
import numpy
from scipy.stats import norm

numpyro.set_host_device_count(4)
az.style.use("arviz-darkgrid")


"""Ean omws parw binary Z me bernoulli(0.5) thn patisame edw giati polles fores nomizei pws epireazei, """
#prepei panta to num_samples na to vazw megalitero apo to sample size
sample_size = 300
num_samples = 1000

def Generate_Observational_Data(sample_size):

    e = dist.Normal(0, 1).sample(jax.random.PRNGKey(93), sample_shape=(sample_size,))
    X = dist.Normal(30, 20).sample(jax.random.PRNGKey(93), sample_shape=(sample_size,)) + e
    Z = dist.Bernoulli(0.5).sample(jax.random.PRNGKey(0), sample_shape=(sample_size,))

    Y = []
    for i in range(len(X)):
        if X[i] <= 25:
            Y.append(0)
        elif 25 < X[i] < 40:
            Y.append(1)
        else:
            Y.append(2)

    data = pd.DataFrame({"Y": np.array(Y), 'X': X, 'Z': Z})

    return data
data = Generate_Observational_Data(sample_size)
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

corr_matrix = data.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
def model1(X,Y,nclasses=3):

    b_X_eta = sample("b_X_eta", Normal(0, 100))

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([nclasses - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    prediction = X * b_X_eta

    numpyro.sample(
        "Y",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),  # probabbilities that we try to predict
        obs=Y
    )

sampler = NUTS(model1)
mcmc = MCMC(sampler, num_samples=num_samples, num_warmup=1000)
mcmc.run(
    jax.random.PRNGKey(93),
    X=data["X"].to_numpy(),
    nclasses=3,
    Y=data['Y'].to_numpy(),
)
mcmc.print_summary()
trace1 = mcmc.get_samples()

def Calculate_Marginal_Likelehood_by_Hand(X,Y,trace):
    arxi = num_samples-sample_size
    a_0 = trace['cutpoints'][:, 0][arxi:num_samples]
    a_1 = trace['cutpoints'][:, 1][arxi:num_samples]
    """Ki edw ta allaxa ola se - SOS giati  prepei a pas kai na diavaseis to documentation to mathimatiko apo th stata 
    gia na deis ti paizei, dld to P(Y<j)= ao+bx autoi to kanoun parametrize kai paizoun orizontas b=-η opote otan kaneis 
    generate ta data esy prepei na exeis kanonia + kai meta paizei mono tou mesa sto paketo to - """
    logit_0 = a_0 - (trace['b_X_eta'][arxi:num_samples] * X)
    logit_1 = a_1 - (trace['b_X_eta'][arxi:num_samples] * X)
    prob_0 = expit(logit_0)
    prob_1 = expit(logit_1) -prob_0
    prob_2 = 1 - expit(logit_1)

    #Calculate Likelihood
    P = 0
    for i in range(len(prob_0)):
        if Y[i] == 0:
            P = P + math.log(prob_0[i])
        elif Y[i] == 1:
            P = P + math.log(prob_1[i])
        else:
            P = P + math.log(prob_2[i])

    Log_Likelihood = P
    fa_0_trace = norm(0, 100).logpdf(numpy.array(a_0))
    fa_1_trace = norm(0, 100).logpdf(numpy.array(a_1))
    fb_trace = norm(0, 100).logpdf(numpy.array(trace['b_X_eta'][arxi:num_samples]))

    f_prior = sum(fa_0_trace)+sum(fa_1_trace)+sum(fb_trace)
    prior = f_prior/(num_samples-arxi)

    return Log_Likelihood+prior

X = data.X
Y = data.Y
Log_likelihood = Calculate_Marginal_Likelehood_by_Hand(X, Y, trace1)
print("Marginal Likelihood without Z:", Log_likelihood)

def model2(X,Y,Z,nclasses=3):

    b_X_eta = sample("b_X_eta", Normal(0, 100))
    b_Z_eta = sample("b_Z_eta", Normal(0, 100))


    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([nclasses - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    prediction = X * b_X_eta + Z * b_Z_eta

    numpyro.sample(
        "Y",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),
        obs=Y
    )

sampler = NUTS(model2)
mcmc = MCMC(sampler, num_samples=num_samples, num_warmup=1000)
mcmc.run(
    jax.random.PRNGKey(93),
    X=data["X"].to_numpy(),
    Z=data["Z"].to_numpy(),
    nclasses=3,
    Y=data['Y'].to_numpy(),
)
mcmc.print_summary()
trace2 = mcmc.get_samples()
def Calculate_Marginal_Likelehood_by_Hand1(X,Y,Z,trace):
    arxi = num_samples-sample_size

    a_0 = trace['cutpoints'][:, 0][arxi:num_samples]
    a_1 = trace['cutpoints'][:, 1][arxi:num_samples]
    """Ki edw ta allaxa ola se - SOS giati  prepei a pas kai na diavaseis to documentation to mathimatiko apo th stata 
    gia na deis ti paizei, dld to P(Y<j)= ao+bx autoi to kanoun parametrize kai paizoun orizontas b=-η opote otan kaneis 
    generate ta data esy prepei na exeis kanonia + kai meta paizei mono tou mesa sto paketo to - """
    logit_0 = a_0 - (trace['b_X_eta'][arxi:num_samples] * X + trace['b_Z_eta'][arxi:num_samples] * Z)
    logit_1 = a_1 - (trace['b_X_eta'][arxi:num_samples] * X + trace['b_Z_eta'][arxi:num_samples] * Z)
    prob_0 = expit(logit_0)
    prob_1 = expit(logit_1) -prob_0
    prob_2 = 1 - expit(logit_1)

    #Calculate Likelihood
    P = 0
    for i in range(len(prob_0)):
        if Y[i] == 0:
            P = P + math.log(prob_0[i])
        elif Y[i] == 1:
            P = P + math.log(prob_1[i])
        else:
            P = P + math.log(prob_2[i])

    Log_Likelihood = P

    fa_0_trace = norm(0, 100).logpdf(numpy.array(a_0))
    fa_1_trace = norm(0, 100).logpdf(numpy.array(a_1))
    fb_trace = norm(0, 100).logpdf(numpy.array(trace['b_X_eta'][arxi:num_samples]))
    fz_trace = norm(0, 100).logpdf(numpy.array(trace['b_Z_eta'][arxi:num_samples]))

    f_prior = sum(fa_0_trace)+sum(fa_1_trace)+sum(fb_trace)+sum(fz_trace)
    prior = f_prior/(num_samples-arxi)

    return Log_Likelihood+prior
X = data.X
Y = data.Y
Z = data.Z
Log_likelihood = Calculate_Marginal_Likelehood_by_Hand1(X, Y, Z, trace2)
print("Marginal Likelihood with Z:", Log_likelihood)
