__author__ = 'brandonkelly'

import pystan
import numpy as np
import os
import matplotlib.pyplot as plt


project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')

ndata = 1000
nbins = 10

concentration = 5.0
bin_probs_parent = np.random.dirichlet(np.ones(nbins))

probs = np.random.dirichlet(concentration * bin_probs_parent, ndata)

total_counts = np.random.negative_binomial(40, 0.75, ndata)
bin_counts = np.zeros((ndata, nbins))

for i in range(ndata):
    bin_counts[i] = np.random.multinomial(total_counts[i], probs[i])

stan_data = {'counts': bin_counts.T, 'ntrain': ndata, 'nbins': nbins}

stan_file = os.path.join(project_dir, 'stan', 'single_component.stan')
fit = pystan.stan(file=stan_file, model_name='single_component_test', data=stan_data, chains=4, iter=1000)

print(fit)

fit.plot()
plt.show()