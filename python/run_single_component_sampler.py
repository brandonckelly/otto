__author__ = 'brandonkelly'

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters import LogNegBinCounts, LogBinProbsGamma, LogConcentration
from bck_mcmc.sampler import Sampler
from bck_mcmc.steps import RobustAdaptiveMetro
import pickle
import multiprocessing


def build_sampler(counts_per_bin, stop_adapting=sys.maxsize):

    negbin = LogNegBinCounts(counts_per_bin.sum(axis=1), 'log-nfailures')

    log_gamma = LogBinProbsGamma(counts_per_bin, 'log-gamma')

    log_conc = LogConcentration(counts_per_bin, 'log-conc')

    log_conc.connect_log_gamma(log_gamma)

    sampler = Sampler()

    nbins = counts_per_bin.shape[1]
    assert nbins == 93

    sampler.add_step(RobustAdaptiveMetro(negbin, stop_adapting_iter=stop_adapting, initial_covar=0.01))
    sampler.add_step(RobustAdaptiveMetro(log_conc, stop_adapting_iter=stop_adapting, initial_covar=0.01))
    sampler.add_step(RobustAdaptiveMetro(log_gamma, stop_adapting_iter=stop_adapting, target_rate=0.15,
                                         initial_covar=0.001 * np.identity(nbins)))

    return sampler


def get_mcmc_samples(sampler):

    r_samples = np.exp(sampler.samples['log-nfailures'])
    c_samples = np.exp(sampler.samples['log-conc'])
    q_samples = np.array(sampler.samples['log-gamma'])
    q_samples = np.exp(q_samples)  # convert from log(gamma) to bin_probs
    q_samples /= q_samples.sum(axis=1)[:, np.newaxis]

    bin_cols = ['bin_probs_' + str(i+1) for i in range(q_samples.shape[1])]
    columns = ['nfailures', 'concentration'] + bin_cols
    samples = pd.DataFrame(np.column_stack((r_samples, c_samples, q_samples)), columns=columns)

    return samples


def run_parallel_chains_helper_(args):
    np.random.seed()
    counts_per_bin, nsamples, burniter, nthin, chain_id = args
    sampler = build_sampler(counts_per_bin, stop_adapting=burniter)

    sampler.run(nsamples, nburn=burniter, nthin=nthin, verbose=True)

    for step in sampler.steps:
        step.report()

    # save the rng seed before pickling
    sampler.rng_state = np.random.get_state()
    with open(os.path.join(data_dir, 'single_component_sampler_' + target + '_chain_' +
                           str(chain_id) + '.pickle'), 'wb') as f:
        pickle.dump(sampler, f)

    samples = get_mcmc_samples(sampler)
    return samples


def run_sampler(counts_per_bin, nsamples, burniter=None, nthin=1, n_jobs=-1, nchains=-1):

    if burniter is None:
        niter = nsamples * nthin
        burniter = niter // 2

    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()
    if nchains < 1:
        nchains = n_jobs

    args_list = [(counts_per_bin, nsamples, burniter, nthin, chain_id) for chain_id in range(nchains)]

    if n_jobs > 1:
        pool = multiprocessing.Pool(n_jobs)
        samples_list = pool.map(run_parallel_chains_helper_, args_list)
    else:
        samples_list = []
        for args in args_list:
            samples = run_parallel_chains_helper_(args)
            samples_list.append(samples)

    samples = pd.concat(samples_list, keys=range(1, nchains + 1), names=['chain', 'iter'])

    return samples


if __name__ == "__main__":

    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')
    plot_dir = os.path.join(project_dir, 'plots')

    nsamples = 5000
    burniter = 5000

    ds_factor = 1  # down-sampling factor

    nfeatures = 93
    columns = ['feat_' + str(i) for i in range(1, 94)]

    train_df = pd.read_hdf(os.path.join(data_dir, 'train_processed.h5'), 'df')

    class_labels = train_df['target'].unique()

    for target in class_labels[-2:]:
        print ''
        print 'Doing class', target
        this_df = train_df.query('target == @target')
        if ds_factor > 1:
            ntrain = len(this_df) // ds_factor
            train_idx = np.random.choice(this_df.index, size=ntrain, replace=False)
            this_df = this_df.loc[train_idx]
        print 'Training with', len(this_df), 'data points...'
        stan_data = {'counts': this_df[columns].values, 'ntrain': len(this_df), 'nbins': len(columns)}

        samples = run_sampler(this_df[columns].values, nsamples, burniter=burniter, nthin=1)

        samples.to_hdf(os.path.join(data_dir, 'single_component_samples_' + target + '.h5'), 'df')