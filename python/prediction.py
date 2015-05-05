__author__ = 'brandonkelly'

import pandas as pd
import numpy as np
from scipy.special import gammaln


def log_sum_of_logs(logx):
    sorted_logx = logx.sort(ascending=False)
    log_sum = sorted_logx[0] + np.log(1.0 + np.sum(np.exp(sorted_logx - sorted_logx[0])))
    return log_sum


def get_loglik_samples_one_class(class_samples, counts, prior_a=1.0, prior_b=1.0):
    """
    Return logarithm of unnormalized class probability for one class for each MCMC sample.

    :param class_samples: MCMC samples for distribution corresponding to this class.
    :param counts: Feature vector for a single data point.
    :param prior_a:
    :param prior_b:
    :return:
    """
    total_counts = counts.sum()
    bin_cols = ['bin_probs_' + str(i + 1) for i in range(93)]

    # compute log-likelihood of each data point, one MCMC sample at a time

    # first calculate component from negative-binomial model for total counts
    loglik = gammaln(prior_b + class_samples['nfailures']) - \
        gammaln(prior_a + prior_b + total_counts + class_samples['nfailures'])
    loglik += gammaln(total_counts + class_samples['nfailures']) - gammaln(class_samples['nfailures'])

    # now add in contribution from dirichlet-multinomial component
    loglik += gammaln(class_samples['concentration']) - \
              gammaln(total_counts + class_samples['concentration'])

    bcounts = class_samples[bin_cols].apply(lambda x: x * class_samples['concentration'])
    bcounts_sum = counts.values + bcounts
    loglik += gammaln(bcounts_sum).sum(axis=1) - gammaln(bcounts).sum(axis=1)

    return loglik


def get_prob_samples_all_classes(samples, counts, class_prior_prob):
    """
    Return class probability for each MCMC sample for a single data point.

    :param samples: MCMC samples for distribution corresponding to each class.
    :param counts: Feature vector for a single data point.
    :param class_prior_prob: Samples for class prior probabilities.
    :return:
    """
    labels = samples.columns.get_level_values(0).unique()
    if np.any(labels != class_prior_prob.columns):
        raise ValueError("class prior probability index must be the same as the first level of the columns of samples")

    prob_samples = pd.DataFrame(index=samples.index, dtype=float, columns=labels)
    prob_samples.columns.name = 'Class Label'

    # draws from class priors, given the training data
    for label in labels:
        # get unnormalized log-probability of test data point being in each class
        loglik_samples_this_class = get_loglik_samples_one_class(samples[label], counts)
        # make sure we add in contribution from prior
        prob_samples[label] = loglik_samples_this_class + np.log(class_prior_prob[label])

    # renormalize so that max(unnormalized probability) = 1, improves numerical stability
    prob_samples = prob_samples.apply(lambda x: x - prob_samples.max(axis=1))
    prob_samples = np.exp(prob_samples)

    # normalize so probabilities sum to one for each MCMC sample
    prob_samples = prob_samples.apply(lambda x: x / prob_samples.sum(axis=1))

    return prob_samples


def classify(samples, counts, compute_rhat=False):
    class_counts = counts.groupby('target').count()[counts.columns[0]]
    fcols = ['feat_' + str(k+1) for k in range(93)]
    counts = counts[fcols]
    class_prior_prob = pd.DataFrame(np.random.dirichlet(class_counts + 1, size=samples.shape[0]), index=samples.index,
                                    columns=class_counts.index)

    class_probs = pd.DataFrame(data=np.empty((len(counts), len(class_counts))), index=counts.index,
                               columns=class_counts.index)

    if compute_rhat:
        rhat = pd.DataFrame(data=np.empty((len(counts), len(class_counts))), index=counts.index,
                            columns=class_counts.index)

    # TODO: compute in parallel
    for idx, x in counts.iterrows():
        print idx
        prob_samples = get_prob_samples_all_classes(samples, x, class_prior_prob)
        class_probs.loc[idx] = prob_samples.mean(axis=0)
        if compute_rhat:
            rhat.loc[idx] = rhat_class_predictions(prob_samples)

    if compute_rhat:
        return class_probs, rhat
    else:
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


def training_misclassification_rate(class_probs, training_data):
    predicted_class = class_probs.idxmax(axis=1)
    correct = predicted_class == training_data['target']
    return np.sum(correct) / float(len(training_data))


def logit(x):
    return np.log(x / (1.0 - x))


def rhat_class_predictions(prob_samples):
    prob_samples = logit(prob_samples)
    # first split the chains in half
    nsamples = np.max(prob_samples.index.get_level_values(1)) + 1
    nchains = len(prob_samples.index.get_level_values(0).unique())
    prob_samples = prob_samples.reset_index()
    prob_samples['chain'] *= 2
    prob_samples['chain'] -= (prob_samples['iter'] < nsamples / 2)
    prob_samples = prob_samples.set_index(['chain', 'iter'])  # reset the index after splitting the chains in half
    nchains *= 2

    chain_average = prob_samples.mean(axis=0, level=0)
    mcmc_average = chain_average.mean()

    between_chain_var = nsamples / (nchains - 1) * ((chain_average - mcmc_average) ** 2).sum()
    within_chain_var = 1.0 / (nsamples - 1.0) * ((prob_samples - chain_average) ** 2).sum(level=0).mean()

    posterior_var = (nsamples - 1.0) / nsamples * within_chain_var + between_chain_var / nsamples

    rhat = np.sqrt(posterior_var / within_chain_var)

    return rhat