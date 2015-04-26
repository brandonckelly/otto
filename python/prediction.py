__author__ = 'brandonkelly'

import pandas as pd
import numpy as np
from scipy.special import gammaln


def log_sum_of_logs(logx):
    sorted_logx = logx.sort(ascending=False)
    log_sum = sorted_logx[0] + np.log(1.0 + np.sum(np.exp(sorted_logx - sorted_logx[0])))
    return log_sum


def get_loglik_samples_one_class(class_samples, counts, prior_a=1.0, prior_b=1.0):
    total_counts = counts.sum(axis=1)
    ndata = len(total_counts)
    unique_tcounts, unique_tcounts_idx = np.unique(total_counts, return_inverse=True)
    bin_cols = ['bin_probs_' + str(i + 1) for i in range(93)]

    loglik_samples = pd.DataFrame(index=class_samples.index, columns=['x_' + str(i+1) for i in range(ndata)],
                                  dtype=float)

    # compute log-likelihood of each data point, one MCMC sample at a time
    for idx, sample in class_samples.iterrows():
        # first calculate component from negative-binomial model for total counts
        loglik = gammaln(prior_b + ndata * sample['nfailures']) - \
            gammaln(prior_a + prior_b + total_counts.sum() + ndata * sample['nfailures'])
        loglik += gammaln(unique_tcounts + sample['nfailures'])[unique_tcounts_idx] - gammaln(sample['nfailures'])

        # now add in contribution from dirichlet-multinomial component
        loglik += gammaln(sample['concentration']) - \
                  gammaln(unique_tcounts + sample['concentration'])[unique_tcounts_idx]

        bcounts_sum = counts + sample['concentration'] * sample[bin_cols]
        uniq_bcsum, u_idx = np.unique(bcounts_sum, return_inverse=True)
        loglik += gammaln(uniq_bcsum)[u_idx].reshape(bcounts_sum.shape).sum(axis=1) - \
                  gammaln(sample['concentration'] * sample[bin_cols]).sum()

        loglik_samples.loc[idx] = loglik

    return loglik_samples


def get_prob_samples_all_classes(samples, counts, prior_counts):
    labels = samples.columns.get_level_values(0).unique()
    column_levels = [labels, ['x_' + str(i+1) for i in range(counts.shape[0])]]
    prob_samples = pd.DataFrame(index=samples.index, dtype=float,
                                columns=pd.MultiIndex.from_product(column_levels, names=['Class Label', 'Data Index']))
    nsamples = samples.shape[0]
    # draws from class priors, given the training data
    class_prior_prob = pd.DataFrame(np.random.dirichlet(prior_counts + 1, size=nsamples), index=samples.index,
                                    columns=labels)
    for label in labels:
        # get unnormalized log-probability of test data point being in each class
        loglik_samples_this_class = get_loglik_samples_one_class(samples[label], counts)
        # make sure we add in contribution from prior
        prob_samples[label] = loglik_samples_this_class + np.log(class_prior_prob[label])

    # renormalize so that max(unnormalized probability) = 1, improves numerical stability
    prob_samples -= prob_samples.max(axis=1, level=1)
    prob_samples = np.exp(prob_samples)
    prob_samples /= prob_samples.sum(axis=1, level=1)  # normalize so probabilities sum to one for each MCMC sample

    return prob_samples


def classify(prob_samples):
    class_probs = prob_samples.mean(axis=0)
    # reshape dataframe to have each class's probability be a separate column
    class_probs = class_probs.reset_index().pivot(index='Data Index', columns='Class Label')
    class_probs['target'] = class_probs
    return class_probs


def training_loss(class_probs, training_data):
    if not np.all(class_probs.index == training_data.index):
        raise ValueError("samples and training_data need to have same index.")

    labels = training_data['target'].unique()
    logloss = 0.0
    for label in labels:
        label_idx = training_data['target'] == label
        logloss += -1.0 * np.sum(np.log(class_probs.loc[label_idx, label]))

    logloss /= len(training_data)
    return logloss


def rhat_class_predictions(predictions):
    pass