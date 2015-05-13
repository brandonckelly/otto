__author__ = 'brandonkelly'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def negbin_sample(nfailure, p, ndata):
    shape = nfailure
    scale = p / (1.0 - p)
    l = np.random.gamma(shape, scale, ndata)
    counts = np.random.poisson(l) + 1
    return counts


def generate_single_component_sample(sample, total_counts_sum, ndata, a=1.0, b=1.0):

    bin_cols = [c for c in sample.index if 'bin_prob' in c]
    nbins = len(bin_cols)
    nfailures = sample['nfailures']

    p = np.random.beta(a + total_counts_sum, b + ndata * nfailures)
    total_counts_draw = negbin_sample(nfailures, p, ndata)

    bin_probs = np.random.dirichlet(sample['concentration'] * sample[bin_cols], ndata)
    bin_counts = np.empty((ndata, nbins))
    for i in range(ndata):
        bin_counts[i] = np.random.multinomial(total_counts_draw[i], bin_probs[i])

    return bin_counts


def generate_predictive_samples(samples, single_sampler=generate_single_component_sample):
    bin_labels = ['bin_' + str(i) for i, c in enumerate(samples.columns) if 'bin_prob' in c]

    # generate posterior draws
    post_data = []
    df_labels = []
    scount = 0
    for idx, sample in samples.iterrows():
        this_sample = single_sampler(sample)
        this_sample_df = pd.DataFrame(this_sample, columns=bin_labels)
        df_labels.append('MCMC ' + str(scount))
        post_data.append(this_sample_df)
        scount += 1

    post_data = pd.concat(post_data, keys=df_labels)

    return post_data


def posterior_predictive_total_counts(post_bin_counts, bin_counts):

    total_counts = bin_counts.sum(axis=1)
    # compare with distribution of measured total counts
    nbins = 50
    pdf, bins = np.histogram(total_counts, bins=nbins)

    # need to reshape to data index along columns and MCMC samples along rows
    post_total_counts = post_bin_counts.sum(axis=1)
    mcmc_indices = post_total_counts.index.get_level_values(0).unique()

    # histogram each of the posterior predictive draws
    post_pdfs = np.empty((len(mcmc_indices), nbins))
    for i, idx in enumerate(mcmc_indices):
        post_pdfs[i], _ = np.histogram(post_total_counts.loc[idx], bins=bins)

    post_pdf_low = np.percentile(post_pdfs, 5, axis=0)
    post_pdf_hi = np.percentile(post_pdfs, 95, axis=0)
    post_pdf_med = np.median(post_pdfs, axis=0)

    xbins = 0.5 * (bins[1:] + bins[:-1])
    plt.fill_between(xbins, post_pdf_hi, post_pdf_low, alpha=0.5, color='DarkOrange')
    plt.plot(xbins, post_pdf_med, '-', lw=2, color='DarkOrange', label='Model')
    plt.plot(xbins, pdf, 'b-', lw=3, label='Data')
    plt.xlabel('Total Counts')

    ax = plt.gca()
    return ax


def posterior_predictive_bin_fracs(post_bin_counts, bin_counts):
    # compare with mean and variance in category fractions
    total_counts = bin_counts.sum(axis=1)
    bin_fracs = bin_counts.apply(lambda x: x / total_counts, axis=0)
    mean_bin_frac = bin_fracs.mean(axis=0)
    std_bin_frac = bin_fracs.std(axis=0)

    # get expected mean and expected variance from MCMC samples
    post_bin_fracs = post_bin_counts.apply(lambda x: x / post_bin_counts.sum(axis=1), axis=0)

    post_bin_mean = post_bin_fracs.mean(axis=0, level=1)
    post_bin_std = post_bin_fracs.std(axis=0, level=1)

    fig = plt.figure()
    ax1 = plt.subplot(211)
    sns.boxplot(post_bin_mean, ax=ax1)
    plt.plot(1 + np.arange(len(mean_bin_frac)), mean_bin_frac, 'ko')
    ax1.set_ylabel('Mean over data')
    ax2 = plt.subplot(212)
    sns.boxplot(post_bin_std, ax=ax2)
    ax2.plot(1 + np.arange(len(std_bin_frac)), std_bin_frac, 'ko')
    ax2.set_ylabel('Std over data')
    ax2.set_xlabel('Bin ID')
    plt.tight_layout()

    return ax1, ax2


def posterior_predictive_rcorrs(post_bin_counts, bin_counts):
    total_counts = bin_counts.sum(axis=1)
    corr_df = bin_counts.apply(lambda x: x / total_counts, axis=0)  # fraction of data occupied in each bin
    corr_df['counts'] = total_counts
    corr_df = corr_df.corr(method='spearman')

    # histogram the correlations
    corr_cols = corr_df.columns.difference(['counts'])  # don't correlate columns with themselves
    nbins = 10
    pdf, bins = np.histogram(corr_df.loc['counts', corr_cols], bins=nbins)

    # now get posterior predictive distribution of correlation histograms
    # index = (mcmc sample, data index), columns = bin labels
    nsamples = len(post_bin_counts.index.get_level_values(0).unique())
    post_total_counts = post_bin_counts.sum(axis=1)
    post_bin_fracs = post_bin_counts.apply(lambda x: x / post_total_counts, axis=0)
    post_bin_fracs['counts'] = post_total_counts

    post_pdfs = np.empty((nsamples, nbins))
    sample_indices = post_bin_fracs.index.get_level_values(0).unique()
    for i, s_idx in enumerate(sample_indices):
        print '   ', i
        corr_this_sample = post_bin_fracs.loc[s_idx].corr(method='spearman')
        corr_cols = corr_this_sample.columns.difference(['counts'])  # don't correlate columns with themselves
        post_pdfs[i], _ = np.histogram(corr_this_sample.loc['counts', corr_cols], bins=bins)

    post_pdf_low = np.percentile(post_pdfs, 5, axis=0)
    post_pdf_hi = np.percentile(post_pdfs, 95, axis=0)
    post_pdf_med = np.median(post_pdfs, axis=0)

    xbins = 0.5 * (bins[1:] + bins[:-1])
    plt.fill_between(xbins, post_pdf_hi, post_pdf_low, alpha=0.5, color='DarkOrange')
    plt.plot(xbins, post_pdf_med, '-', lw=2, color='DarkOrange', label='Model')
    plt.plot(xbins, pdf, 'b-', lw=3, label='Data')
    plt.xlabel('Correlation between Total Counts and Bin Fractions')

    ax = plt.gca()
    return ax


def posterior_predictive_bin_corrs(post_bin_counts, bin_counts):
    total_counts = bin_counts.sum(axis=1)
    corr_df = bin_counts.apply(lambda x: x / total_counts, axis=0)  # fraction of data occupied in each bin
    corr_df = corr_df.corr(method='spearman')

    # histogram the correlations
    diag_idx = np.diag_indices_from(corr_df)
    corr_df = corr_df.values
    corr_df[diag_idx] = 2
    off_diag_idx = corr_df < 1.5  # don't correlate columns with themselves

    nbins = 100
    pdf, bins = np.histogram(corr_df[off_diag_idx], bins=nbins)

    # now get posterior predictive distribution of correlation histograms
    # index = (mcmc sample, data index), columns = bin labels
    nsamples = len(post_bin_counts.index.get_level_values(0).unique())
    post_total_counts = post_bin_counts.sum(axis=1)
    post_bin_fracs = post_bin_counts.apply(lambda x: x / post_total_counts, axis=0)
    post_bin_fracs['counts'] = post_total_counts

    post_pdfs = np.empty((nsamples, nbins))
    sample_indices = post_bin_fracs.index.get_level_values(0).unique()
    for i, s_idx in enumerate(sample_indices):
        print '      ', i
        corr_this_sample = post_bin_fracs.loc[s_idx].corr(method='spearman').values
        post_pdfs[i], _ = np.histogram(corr_this_sample[off_diag_idx], bins=bins)

    post_pdf_low = np.percentile(post_pdfs, 5, axis=0)
    post_pdf_hi = np.percentile(post_pdfs, 95, axis=0)
    post_pdf_med = np.median(post_pdfs, axis=0)

    xbins = 0.5 * (bins[1:] + bins[:-1])
    plt.fill_between(xbins, post_pdf_hi, post_pdf_low, alpha=0.5, color='DarkOrange')
    plt.plot(xbins, post_pdf_med, '-', lw=2, color='DarkOrange', label='Model')
    plt.plot(xbins, pdf, 'b-', lw=3, label='Data')
    plt.xlabel('Correlations Among Bin Fractions')

    ax = plt.gca()
    return ax


def posterior_predictive_check(samples, bin_counts, class_label, nsamples=None):

    if nsamples is None:
        mcmc_samples = samples.copy()
    else:
        mcmc_idx = np.linspace(0, len(samples), nsamples, endpoint=False, dtype=int)
        mcmc_samples = samples.iloc[mcmc_idx]

    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    plot_dir = os.path.join(project_dir, 'plots', 'single_component')

    total_counts = bin_counts.sum(axis=1)
    ndata = len(total_counts)
    total_counts_sum = np.sum(total_counts)

    single_sampler = lambda x: generate_single_component_sample(x, total_counts_sum, ndata)

    print 'Generating predictive samples...'
    data_samples = generate_predictive_samples(mcmc_samples, single_sampler=single_sampler)

    print 'Comparing with histogram of total counts...'
    ax = posterior_predictive_total_counts(data_samples, bin_counts)
    ax.set_title(class_label)
    plt.savefig(os.path.join(plot_dir, 'post_check_total_counts_' + class_label + '.png'))
    # plt.show()
    plt.close()

    print 'Comparing with first and second moments of bin fractions...'
    ax1, ax2 = posterior_predictive_bin_fracs(data_samples, bin_counts)
    ax1.set_title(class_label)
    plt.savefig(os.path.join(plot_dir, 'post_check_bin_fracs_' + class_label + '.png'))
    # plt.show()
    plt.close()

    print 'Comparing with correlations between total counts and bin fractions...'
    ax = posterior_predictive_rcorrs(data_samples, bin_counts)
    ax.set_title(class_label)
    plt.savefig(os.path.join(plot_dir, 'post_check_counts_corrs_' + class_label + '.png'))
    # plt.show()
    plt.close()

    print 'Comparing with correlations among bin fractions...'
    ax = posterior_predictive_bin_corrs(data_samples, bin_counts)
    ax.set_title(class_label)
    plt.savefig(os.path.join(plot_dir, 'post_check_bin_corrs_' + class_label + '.png'))
    # plt.show()
    plt.close()
