__author__ = 'brandonkelly'


import pystan
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.special import gammaln
import cPickle


def nfailure_posterior(n, a=1.0, b=1.0):
    rgrid = np.linspace(0.0, 100.0, 10000)
    logpost = gammaln(a + np.sum(n)) + gammaln(b + len(n) * rgrid) - gammaln(a + b + np.sum(n) + len(n) * rgrid)
    for i in range(len(n)):
        logpost += gammaln(n[i] + rgrid) - gammaln(n[i] + 1.0) - gammaln(rgrid)

    # refine the grid
    max_idx = np.argmax(logpost)
    thresh = logpost[max_idx] - np.log(200)
    lowest_idx = np.sum(logpost[:max_idx] < thresh) + 1
    new_low = rgrid[lowest_idx]
    highest_idx = np.sum(logpost[max_idx:] >= thresh) + max_idx - 1
    new_high = rgrid[highest_idx]
    rgrid = np.linspace(new_low, new_high, 1000)
    logpost = gammaln(a + np.sum(n)) + gammaln(b + len(n) * rgrid) - gammaln(a + b + np.sum(n) + len(n) * rgrid)
    for i in range(len(n)):
        logpost += gammaln(n[i] + rgrid) - gammaln(n[i] + 1.0) - gammaln(rgrid)

    return rgrid, logpost


def get_initial_values(counts, nchains=4):

    total_counts = counts.sum(axis=1)
    fractions = counts / total_counts[:, np.newaxis].astype(float)

    # initial guesses for dirichlet-multinomial component
    fraction = np.mean(fractions, axis=0)
    fraction /= fraction.sum()

    concentration = fraction * (1.0 - fraction) / fractions.var(axis=0) - 1.0
    concentration[concentration <= 0] = 1.0

    # initial guesses for negative binomial component
    rgrid, log_marginal = nfailure_posterior(total_counts)
    nf_marginal = np.exp(log_marginal - log_marginal.max())
    nf_marginal /= nf_marginal.sum()

    iguesses = []
    for c in range(nchains):
        if c < 10:
            bin_prob_guess = fraction
        else:
            bin_prob_guess = np.random.dirichlet(counts.sum(axis=0) / 10.0)
        concentration_guess = np.random.lognormal(np.log(np.median(concentration)), 1e-6)
        # negbin_failures_guess = np.random.choice(rgrid, p=nf_marginal)
        negbin_failures_guess = rgrid[np.argmax(nf_marginal)]
        iguesses.append({'bin_probs': bin_prob_guess,
                         'concentration': concentration_guess,
                         'negbin_nfailures': negbin_failures_guess})

    return iguesses


def run_stan(stan_data, target):

    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')

    iguesses = get_initial_values(stan_data['counts'])

    stan_file = os.path.join(project_dir, 'stan', 'single_component.stan')
    fit = pystan.stan(file=stan_file, model_name='single_component_' + target, data=stan_data,
                      chains=4, iter=400, warmup=200, init=iguesses)

    # dump the MCMC samples to an HDF file
    samples = fit.extract()

    cnames = ['concentration', 'negbin_nfailures']
    nbins = stan_data['counts'].shape[1]
    cnames += ['bin_prob_' + str(i + 1) for i in range(nbins)]

    raw_samples = np.column_stack((samples['concentration'], samples['negbin_nfailures'], samples['bin_probs']))
    samples = pd.DataFrame(data=raw_samples, columns=cnames)
    samples.to_hdf(os.path.join(data_dir, 'single_component_' + target + '_samples.h5'), 'df')

    # dump the stan results to a text file
    with open(os.path.join(data_dir, 'single_component_' + target + '_stan_summary.txt'), 'w') as f:
        print >> f, fit

    # make plots of the stan results
    plot_dir = os.path.join(project_dir, 'plots')
    fit.plot()
    plt.savefig(os.path.join(plot_dir, 'single_component_' + target + '_trace.png'))

    return


if __name__ == "__main__":

    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')

    nfeatures = 93
    columns = ['feat_' + str(i) for i in range(1, 94)]

    train_df = pd.read_hdf(os.path.join(data_dir, 'train_processed.h5'), 'df')

    class_labels = train_df['target'].unique()

    for target in class_labels:
        print ''
        print 'Doing class', target
        this_df = train_df[train_df['target'] == target]
        stan_data = {'counts': this_df[columns].values, 'ntrain': len(this_df), 'nbins': len(columns)}

        run_stan(stan_data, target)