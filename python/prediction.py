__author__ = 'brandonkelly'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


def negbin_sample(nfailure, p, ndata):
    shape = nfailure
    scale = p / (1.0 - p)
    l = np.random.gamma(shape, scale, ndata)
    counts = np.random.poisson(l)
    return counts


def generate_single_component_sample(sample, total_counts, ndata, a=1.0, b=1.0):

    bin_cols = [c for c in sample.index if 'bin_prob' in c]
    nbins = len(bin_cols)
    nfailures = sample['nfailures']

    p = np.random.beta(a + total_counts, b + ndata * nfailures)
    total_counts_draw = np.empty(ndata)
    for i in range(ndata):
        total_counts_draw[i] = negbin_sample(nfailures, p, ndata)

    bin_probs = np.random.dirichlet(sample['concentration'] * sample[bin_cols], ndata)
    bin_counts = np.empty((ndata, nbins))
    for i in range(ndata):
        bin_counts[i] = np.random.multinomial(total_counts_draw[i], bin_probs)

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
    pdf, bins = np.histogram(total_counts, bins=50)

    # need to reshape to data index along rows and MCMC samples along columns
    post_pdfs = post_bin_counts.sum(axis=1).reset_index(level=0)

    post_pdf_low = post_pdfs.quantile(0.05, axis=1)
    post_pdf_hi = post_pdfs.quantile(0.95, axis=1)
    post_pdf_med = post_pdfs.median(axis=1)

    plt.fill_between(bins, post_pdf_hi, post_pdf_low, alpha=0.25, color='DarkOrange')
    plt.plot(bins, post_pdf_med, '-', lw=2, color='DarkOrange', label='Model')
    plt.plot(bins, pdf, 'b-', lw=3, label='Data')


def posterior_predictive_bin_fracs(post_bin_counts, bin_counts):
    # compare with mean and variance in category fractions
    total_counts = bin_counts.sum(axis=1)
    bin_fracs = bin_counts / total_counts[:, np.newaxis]
    mean_bin_frac = bin_fracs.mean(axis=0)
    std_bin_frac = bin_fracs.std(axis=0)

    # get expected mean and expected variance from MCMC samples
    post_bin_fracs = post_bin_counts.apply(lambda x: x / post_bin_counts.sum(axis=1), axis=0)

    post_bin_mean = post_bin_fracs.mean(axis=0, level=1)
    post_bin_std = post_bin_fracs.std(axis=0, level=1)

    fig = plt.figure()
    ax1 = plt.subplot(211)
    sns.boxplot(post_bin_mean, ax=ax1)
    plt.plot(1 + len(post_bin_counts), mean_bin_frac, 'ko')
    ax1.set_ylabel('Mean over data')
    ax2 = plt.subplot(212)
    sns.boxplot(post_bin_std, ax=ax2)
    ax2.plot(1 + len(post_bin_counts), std_bin_frac, 'ko')
    ax2.set_ylabel('Std over data')
    ax2.set_xlabel('Bin ID')
    plt.tight_layout()


def posterior_predictive_rcorrs(post_bin_counts, bin_counts):
    pass


def posterior_predictive_bin_corrs(post_bin_counts, bin_counts):
    pass


def posterior_predictive_check(samples, bin_counts, class_label):

    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    plot_dir = os.path.join(project_dir, 'plots', 'single_component')

    data_samples = generate_predictive_samples(samples, single_sampler=generate_predictive_samples)

    posterior_predictive_total_counts(data_samples, bin_counts)
    plt.savefig(os.path.join(plot_dir, 'post_check_total_counts_' + class_label + '.png'))
    plt.show()
    plt.close()

    posterior_predictive_bin_fracs(data_samples, bin_counts)
    plt.savefig(os.path.join(plot_dir, 'post_check_bin_fracs_' + class_label + '.png'))
    plt.show()
    plt.close()

    posterior_predictive_rcorrs(data_samples, bin_counts)
    plt.savefig(os.path.join(plot_dir, 'post_check_rcorrs_' + class_label + '.png'))
    plt.show()
    plt.close()

    posterior_predictive_bin_corrs(data_samples, bin_counts)
    plt.savefig(os.path.join(plot_dir, 'post_check_corrs_' + class_label + '.png'))
    plt.show()
    plt.close()


def classify(samples, test_counts):
    pass


def training_misclassification_rate(samples, training_counts):
    pass


def rhat_class_predictions(predictions):
    pass