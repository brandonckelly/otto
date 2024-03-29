__author__ = 'brandonkelly'

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters import MixLogNegBinPars, MixtureComponents, LogBinProbsAlpha, PriorCovar, PriorMu, PriorVar, \
    StickWeight, DPconcentration, alpha_inverse_transform, alpha_transform
from bck_mcmc.sampler import Sampler
from bck_mcmc.steps import RobustAdaptiveMetro, GibbsStep
import pickle
import multiprocessing


def build_sampler(counts_per_bin, ncomponents, stop_adapting=sys.maxsize):
    nbins = counts_per_bin.shape[1]
    assert nbins == 93

    sampler = Sampler()
    component_labels = MixtureComponents('component_label', counts_per_bin, track=False)

    # prior parameters
    prior_negbin_mean = PriorMu('negbin-prior-mean', np.array([np.log(40), np.log(2.0), np.log(2.0)]),
                                np.diag([2.0, 0.5 ** 2, 0.5 ** 2]),
                                track=False)
    prior_negbin_covar = PriorCovar('negbin-prior-covar', 5, np.diag([1.0, 0.1, 0.1]), track=False)
    prior_alpha_mean = PriorMu('alpha-prior-mean', np.array([np.log(nbins)] + (nbins - 1) * [0]),
                               np.array([2.0 ** 2] + (nbins - 1) * [2.0 ** 2]), track=False)
    prior_alpha_var = PriorVar('alpha-prior-var', 1 * np.ones(nbins), 2.0 * np.ones(nbins),
                               track=False)
    prior_negbin_mean.child_var = prior_negbin_covar
    prior_negbin_covar.child_mean = prior_negbin_mean
    prior_alpha_mean.child_var = prior_alpha_var
    prior_alpha_var.child_mean = prior_alpha_mean

    stick = StickWeight('stick-weights')
    component_labels.stick_weights = stick
    dp_conc = DPconcentration('dp-concentration', 1.0, 1.0)
    dp_conc.stick_weights = stick
    stick.concentration = dp_conc
    stick.components = component_labels

    # add steps for parameters to be initialized first
    sampler.add_step(GibbsStep(dp_conc))
    sampler.add_step(GibbsStep(component_labels))
    sampler.add_step(GibbsStep(stick))

    for k in range(ncomponents):
        negbin = MixLogNegBinPars(counts_per_bin.sum(axis=1), 'negbin-' + str(k), k)
        alpha_inverse = LogBinProbsAlpha(counts_per_bin, 'alpha-' + str(k), k)

        # connect the parameters
        negbin.connect_prior(prior_negbin_mean, prior_negbin_covar)
        alpha_inverse.connect_prior(prior_alpha_mean, prior_alpha_var)
        component_labels.add_component(negbin, alpha_inverse, k)

        # add steps for this component
        sampler.add_step(RobustAdaptiveMetro(negbin, stop_adapting_iter=stop_adapting,
                                             initial_covar=0.01 * np.identity(3), target_rate=0.3))
        sampler.add_step(RobustAdaptiveMetro(alpha_inverse, stop_adapting_iter=stop_adapting,
                                             initial_covar=0.001 * np.identity(nbins), target_rate=0.15))

    # steps for prior
    sampler.add_step(GibbsStep(prior_alpha_mean))  # means need to be initialized last
    sampler.add_step(GibbsStep(prior_alpha_var))
    sampler.add_step(GibbsStep(prior_negbin_mean))
    sampler.add_step(GibbsStep(prior_negbin_covar))

    return sampler


def get_mcmc_samples(sampler, ncomponents):

    stick_samples = sampler.samples['stick-weights']
    columns = [('Stick Weights', str(k)) for k in range(ncomponents-1)]
    dp_conc_samples = sampler.samples['dp-concentration']
    columns.append(('DP Concentration', 'DP Concentration'))

    data_columns = [stick_samples, dp_conc_samples]

    for k in range(ncomponents):
        negbin_samples_k = np.exp(np.array(sampler.samples['negbin-' + str(k)]))
        data_columns.append(negbin_samples_k)

        alpha_samples_k = np.array(sampler.samples['alpha-' + str(k)])
        for i in range(alpha_samples_k.shape[0]):
            alpha_samples_k[i] = alpha_inverse_transform(alpha_samples_k[i])
        data_columns.append(alpha_samples_k)

        negbin_cols = ['nfailure', 'beta_a', 'beta_b']
        negbin_cols = [('Component ' + str(k), c) for c in negbin_cols]
        columns.extend(negbin_cols)

        alpha_cols = ['alpha_' + str(j + 1) for j in range(alpha_samples_k.shape[1])]
        alpha_cols = [('Component ' + str(k), c) for c in alpha_cols]
        columns.extend(alpha_cols)

    samples = pd.DataFrame(np.column_stack(data_columns), columns=pd.MultiIndex.from_tuples(columns))

    return samples


def run_parallel_chains_helper_(args):
    np.random.seed()
    counts_per_bin, ncomponents, nsamples, burniter, nthin, chain_id, data_dir, target = args
    sampler = build_sampler(counts_per_bin, ncomponents)

    sampler.run(nsamples, nburn=burniter, nthin=nthin, verbose=True)

    for step in sampler.steps:
        if isinstance(step, RobustAdaptiveMetro):
            step.report()

    # save the rng seed before pickling
    sampler.rng_state = np.random.get_state()
    with open(os.path.join(data_dir, 'multi_component_sampler_' + target + '_chain_' +
                           str(chain_id) + '.pickle'), 'wb') as f:
        pickle.dump(sampler, f)

    samples = get_mcmc_samples(sampler, ncomponents)
    return samples


def run_sampler(counts_per_bin, nsamples, ncomponents, target, burniter=None, nthin=1, n_jobs=-1, nchains=-1):
    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')

    if burniter is None:
        niter = nsamples * nthin
        burniter = niter // 2

    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()
    if nchains < 1:
        nchains = n_jobs

    args_list = [(counts_per_bin, ncomponents, nsamples, burniter, nthin, chain_id, data_dir, target)
                 for chain_id in range(nchains)]

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


def main(ncomponents, nsamples, burniter):

    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')
    plot_dir = os.path.join(project_dir, 'plots')

    ntrain = sys.maxint

    nfeatures = 93
    columns = ['feat_' + str(i) for i in range(1, 94)]

    train_df = pd.read_hdf(os.path.join(data_dir, 'train_processed.h5'), 'df')

    class_labels = train_df['target'].unique()

    for target in class_labels:
        print ''
        print 'Doing class', target
        this_df = train_df.query('target == @target')

        if ntrain < len(this_df):
            train_idx = np.random.choice(this_df.index, size=ntrain, replace=False)
            this_df = this_df.loc[train_idx]

        print 'Training with', len(this_df), 'data points...'

        samples = run_sampler(this_df[columns].values, nsamples, ncomponents, target,
                              burniter=burniter, nthin=1, n_jobs=-1)

        samples.to_hdf(os.path.join(data_dir, 'multi_component_samples_' + target + '.h5'), 'df')

if __name__ == "__main__":

    ncomponents = 15
    nsamples = 100
    burniter = 50

    main(ncomponents, nsamples, burniter)