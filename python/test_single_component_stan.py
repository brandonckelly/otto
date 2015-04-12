__author__ = 'brandonkelly'

import pystan
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.special import gammaln


project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')

ndata = 1000
nbins = 10

concentration = 5.0
bin_probs_parent = np.random.dirichlet(np.ones(nbins))

probs = np.random.dirichlet(concentration * bin_probs_parent, ndata)

success_prob = 0.75
nfailures = 40
total_counts = np.random.negative_binomial(nfailures, success_prob, ndata)
bin_counts = np.zeros((ndata, nbins), dtype=np.int)

for i in range(ndata):
    bin_counts[i] = np.random.multinomial(total_counts[i], probs[i])


def nfailure_posterior(n, a=1.0, b=1.0):
    rgrid = np.arange(1, 100)
    logpost = gammaln(a + np.sum(n)) + gammaln(b + len(n) * rgrid) - gammaln(a + b + np.sum(n) + len(n) * rgrid)
    for i in range(len(n)):
        logpost += gammaln(n[i] + rgrid) - gammaln(n[i] + 1.0) - gammaln(rgrid)

    return logpost

# logpost = nfailure_posterior(total_counts, a=1.0, b=1.0)
# plt.plot(np.exp(logpost - logpost.max()), label='a=1, b=1')
# logpost = nfailure_posterior(total_counts, a=0.1, b=1.0)
# plt.plot(np.exp(logpost - logpost.max()), label='a=0.1, b=1')
# logpost = nfailure_posterior(total_counts, a=1.0, b=0.1)
# plt.plot(np.exp(logpost - logpost.max()), label='a=1, b=0.1')
# logpost = nfailure_posterior(total_counts, a=5.0, b=5.0)
# plt.plot(np.exp(logpost - logpost.max()), label='a=5, b=5')
# plt.legend(loc='best')
# plt.show()

stan_data = {'counts': bin_counts, 'ntrain': ndata, 'nbins': nbins}

stan_file = os.path.join(project_dir, 'stan', 'single_component.stan')
fit = pystan.stan(file=stan_file, model_name='single_component_test', data=stan_data, chains=4, iter=1000)

print(fit)

fit.plot()
plt.show()

samples = fit.extract()

cnames = ['concentration', 'negbin_nfailures']
for i in range(nbins):
    cnames.append('bin_prob_' + str(i))

raw_samples = np.column_stack((samples['concentration'], samples['negbin_nfailures'], samples['bin_probs']))
samples = pd.DataFrame(data=raw_samples, columns=cnames)

sns.violinplot(samples)
plt.yscale('log')
plt.plot(range(1, 1 + samples.shape[1]), [concentration, nfailures] + list(bin_probs_parent), 'ko')
plt.show()

sns.jointplot('negbin_nfailures', 'concentration', data=samples, kind='kde')
plt.show()