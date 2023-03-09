"""Inspired by https://www.youtube.com/watch?v=KxrrEqYfhF0&ab_channel=PyData """
import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticket
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import scipy
import scipy.stats as stats
import seaborn as sns
from numpyro import handlers
from numpyro.infer.reparam import TransformReparam
numpyro.set_host_device_count(4)
az.style.use("arviz-darkgrid")

url = "https://stats.idre.ucla.edu/stat/data/ologit.dta"
df = pd.read_stata(url)

print(df.head(5))
apply_to_index = {'unlikely': 0, 'somewhat likely': 1, 'very likely': 2}
df['apply'].value_counts().reindex(apply_to_index).plot.barh()
# plt.show()

def ordered_logistic_regression(pared,public,n_apply_levels, apply=None):
    coefficient_neither = numpyro.sample(
        "coefficient_neither",
        dist.Normal(0, 1),
    )
    coefficient_pared = numpyro.sample(
        "coefficient_pared",
        dist.Normal(0, 1),
    )
    coefficient_public = numpyro.sample(
        "coefficient_public",
        dist.Normal(0, 1),
    )
    coefficient_pared_and_public = numpyro.sample(
        "coefficient_pared_and_public",
        dist.Normal(0, 1),
    )
    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([n_apply_levels-1]),
            dist.transforms.OrderedTransform(),
        ),
    )

    prediction = (
            coefficient_neither * np.where((pared == 0) & (public == 0), 1, 0)
            + coefficient_pared * np.where((pared == 1) & (public == 0), 1, 0)
            + coefficient_public * np.where((pared == 0) & (public == 1), 1, 0)
            + coefficient_pared_and_public * np.where((pared == 1) & (public == 1), 1, 0)
    )

    #Apo edw ws edw mporw na to kanw me ena function OrderedLogistic
    # logits = cutpoints - prediction[:, jnp.newaxis]
    # cumulative_probs = jnp.pad(
    #     jax.scipy.special.expit(logits),
    #     pad_width=((0, 0), (1, 1)),
    #     constant_values=(0, 1),
    # )
    # probs = numpyro.deterministic("probs", jnp.diff(cumulative_probs))
    #
    # numpyro.sample(
    #     "apply",
    #     dist.Categorical(probs=probs), #probabbilities that we try to predict
    #     obs=apply
    # )
    """Pws mporeis na antikatastiseis to panw me ena function"""
    numpyro.sample(
        "apply",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),  # probabbilities that we try to predict
        obs=apply
    )



"""Prior predictive simulattion kai to kanoume gia 4 atoma, xrisimopoiei mono ta priors pou edwsa"""
# prior_pred = numpyro.infer.Predictive(ordered_logistic_regression, num_samples=100)
# prior_predictions = prior_pred(
#     jax.random.PRNGKey(93),
#     pared=np.array([0, 0, 1, 1]),
#     public=np.array([0, 1, 0, 1]),
#     n_apply_levels=3,
# )
#
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
# ax = ax.flatten()
# rows = {
#     0: "No pared, no public",
#     1: "Pared",
#     2: "Public",
#     3: "Pared and public"
# }
#
# for row, description in rows.items():
#     prior_df = pd.DataFrame(
#         prior_predictions["probs"][:, row, :], columns=apply_to_index
#     )
#     sns.kdeplot(data=prior_df, ax=ax[row], cut=0)
#     ax[row].set_title(description)
#
# plt.show()

"""MCMC"""

sampler = numpyro.infer.NUTS(ordered_logistic_regression)
mcmc = numpyro.infer.MCMC(sampler, num_chains=4, num_samples=1000, num_warmup=1000)
mcmc.run(
    jax.random.PRNGKey(93),
    pared=df["pared"].to_numpy(),
    public=df["public"].to_numpy,
    n_apply_levels=3,
    apply=df['apply'].map(apply_to_index).to_numpy(),
)
mcmc.print_summary()
print(mcmc.get_samples()['coefficient_neither'])

# post_pred = numpyro.infer.Predictive(ordered_logistic_regression, mcmc.get_samples())
# post_predictions = post_pred(
#     jax.random.PRNGKey(93),
#     pared=np.array([0, 0, 1, 1]),
#     public=np.array([0, 1, 0, 1]),
#     n_apply_levels=3
# )
#
# fig, ax = plt.subplots(figsize=(14, 8))
# for coefficient in ['coefficient_pared', 'coefficient_public', 'coefficient_pared_and_public', 'coefficient_neither']:
#     sns.kdeplot(mcmc.get_samples()[coefficient], ax=ax, label=coefficient)
# ax.legend()
# plt.show()
#
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 6), sharex=True, sharey=True)
# ax = ax.flatten()
# rows = {
#     0: "No pared, no public",
#     1: "pared",
#     2: "public",
#     3: "Pared and public"
# }
#
# for row, description in rows.items():
#     prior_df = pd.DataFrame(
#         post_predictions["probs"][:, row, :], columns=apply_to_index
#     )
#     sns.kdeplot(data=prior_df, ax=ax[row], cut=8)
#     ax[row].set_title(description)
# plt.show()