"""Inspired by https://www.youtube.com/watch?v=KxrrEqYfhF0&ab_channel=PyData """
import arviz as az
import jax
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro import handlers
numpyro.set_host_device_count(4)
az.style.use("arviz-darkgrid")

url = "https://stats.idre.ucla.edu/stat/data/ologit.dta"
df = pd.read_stata(url)

apply_to_index = {'unlikely': 0, 'somewhat likely': 1, 'very likely': 2}
df['apply'].value_counts().reindex(apply_to_index).plot.barh()

num_samples = 400
df = df.head(num_samples)
rng_key = jax.random.PRNGKey(93)

def ordered_logistic_regression(pr,pl,g,n_apply_levels, apply=None):
    pared = numpyro.sample(
        "pared",
        dist.Normal(0, 100),
    )
    public = numpyro.sample(
        "public",
        dist.Normal(0, 100),
    )
    gpa = numpyro.sample(
        "gpa",
        dist.Normal(0, 100),
    )

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([n_apply_levels-1]),
            dist.transforms.OrderedTransform(),
        ),
    )

    prediction = (pr * pared + pl * public + g * gpa)

    """Pws mporeis na antikatastiseis to panw me ena function"""
    numpyro.sample(
        "apply",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),  # probabbilities that we try to predict
        obs=apply
    )
def log_likelihood(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['apply']
    return sum(obs_node['fn'].log_prob(obs_node['value']))

sampler = numpyro.infer.NUTS(ordered_logistic_regression)
mcmc = numpyro.infer.MCMC(sampler, num_samples=num_samples, num_warmup=1000)
mcmc.run(
    jax.random.PRNGKey(93),
    pr=df["pared"].to_numpy(),
    pl=df["public"].to_numpy(),
    g=df["gpa"].to_numpy(),
    n_apply_levels=3,
    apply=df["apply"].map(apply_to_index).to_numpy(),
)
trace = mcmc.get_samples()
mcmc.print_summary()
pared = df["pared"].to_numpy()
public = df["public"].to_numpy()
gpa = df["gpa"].to_numpy()
n_apply_levels = 3
Log_Likelihood = log_likelihood(jax.random.PRNGKey(93), trace, ordered_logistic_regression, pared, public, gpa,
                                n_apply_levels)
print("Log Likelihood : ", Log_Likelihood)

def ordered_logistic_regression1(pr,pl,n_apply_levels, apply=None):
    pared = numpyro.sample(
        "pared",
        dist.Normal(0, 100),
    )
    public = numpyro.sample(
        "public",
        dist.Normal(0, 100),
    )

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([n_apply_levels-1]),
            dist.transforms.OrderedTransform(),
        ),
    )

    prediction = (pr * pared + pl * public)

    """Pws mporeis na antikatastiseis to panw me ena function"""
    numpyro.sample(
        "apply",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),  # probabbilities that we try to predict
        obs=apply
    )
def log_likelihood(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['apply']
    return sum(obs_node['fn'].log_prob(obs_node['value']))

sampler = numpyro.infer.NUTS(ordered_logistic_regression1)
mcmc = numpyro.infer.MCMC(sampler, num_samples=num_samples, num_warmup=1000)
mcmc.run(
    jax.random.PRNGKey(93),
    pr=df["pared"].to_numpy(),
    pl=df["public"].to_numpy(),
    n_apply_levels=3,
    apply=df["apply"].map(apply_to_index).to_numpy(),
)
trace = mcmc.get_samples()

pared = df["pared"].to_numpy()
public = df["public"].to_numpy()
n_apply_levels = 3
Log_Likelihood = log_likelihood(jax.random.PRNGKey(93), trace, ordered_logistic_regression1, pared, public,
                                n_apply_levels)
print("Log Likelihood : ", Log_Likelihood)