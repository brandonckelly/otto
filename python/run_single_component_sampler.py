__author__ = 'brandonkelly'

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters import LogNegBinCounts, LogBinProbsGamma, LogConcentration
from bck_mcmc.sampler import Sampler
from bck_mcmc.steps import RobustAdaptiveMetro
import cPickle
import multiprocessing


def build_sampler(counts_per_bin, stop_adapting=sys.maxsize):

    negbin = LogNegBinCounts(counts_per_bin.sum(axis=1), 'log-nfailures')

    log_gamma = LogBinProbsGamma(counts_per_bin, 'log-gamma')

    log_conc = LogConcentration(counts_per_bin, 'log-conc')

    log_conc.connect_log_gamma(log_gamma)

    sampler = Sampler()

    sampler.add_step(RobustAdaptiveMetro(negbin, stop_adapting_iter=stop_adapting, initial_covar=0.01))
    sampler.add_step(RobustAdaptiveMetro(log_conc, stop_adapting_iter=stop_adapting, initial_covar=0.01))
    sampler.add_step(RobustAdaptiveMetro(log_gamma, stop_adapting_iter=stop_adapting,
                                         initial_covar=0.01 * np.identity(counts_per_bin.shape[1])))

    return sampler


def run_sampler(counts_per_bin, nsamples, burniter=None, nthin=1):

    if burniter is None:
        niter = nsamples * nthin
        burniter = niter // 2

    sampler = build_sampler(counts_per_bin, stop_adapting=burniter)

    sampler.run(nsamples, nburn=burniter, nthin=nthin, verbose=True)

    return sampler


def get_mcmc_samples(sampler):
    samples = sampler.get_samples()

    r_samples = np.exp(sampler.samples['log-nfailures'])
    c_samples = np.exp(sampler.samples['log-conc'])
    q_samples = np.array(sampler.samples['log-gamma'])
    q_samples = np.exp(q_samples)  # convert from log(gamma) to bin_probs
    q_samples /= q_samples.sum(axis=1)[:, np.newaxis]

    bin_cols = ['bin_probs_' + str(i+1) for i in range(q_samples.shape[1])]
    columns = ['nfailures', 'concentration'] + bin_cols
    samples = pd.DataFrame(np.column_stack((r_samples, c_samples, q_samples)), columns=columns)

    return samples


if __name__ == "__main__":

    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')
    plot_dir = os.path.join(project_dir, 'plots')

    nsamples = 1000
    burniter = 1000

    nfeatures = 93
    columns = ['feat_' + str(i) for i in range(1, 94)]

    train_df = pd.read_hdf(os.path.join(data_dir, 'train_processed.h5'), 'df')

    class_labels = train_df['target'].unique()

    for target in class_labels:
        print ''
        print 'Doing class', target
        this_df = train_df[train_df['target'] == target]
        stan_data = {'counts': this_df[columns].values, 'ntrain': len(this_df), 'nbins': len(columns)}

        sampler = run_sampler(this_df[columns].values, nsamples, burniter=burniter, nthin=1)

        samples = sampler.get_samples()
        sampler.to_hdf(os.path.join(data_dir, 'single_component_samples_' + target + '.h5', 'df'))

        # should probably save the rng seed before pickling
        with open(os.path.join(data_dir, 'single_component_sampler_' + target + '.pickle'), 'wb') as f:
            cPickle.dump(sampler, f)

        # TODO: make plots, add multiple chains