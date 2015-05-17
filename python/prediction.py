__author__ = 'brandonkelly'

import pandas as pd
import numpy as np
from scipy.special import gammaln
import multiprocessing
from parameters import alpha_inverse_transform


def log_sum_of_logs(logx):
    m = logx.max()
    y = logx - m
    log_sum = m + np.log(np.sum(np.exp(y)))

    return log_sum


def get_loglik_samples_one_class_multi_component(class_samples, counts):
    """
    Return logarithm of unnormalized class probability for one class for each MCMC sample.

    :param class_samples: MCMC samples for distribution corresponding to this class.
    :param counts: Feature vector for a single data point.

    :return:
    """

    # get the cluster weights\
    stick_weights = class_samples['Stick Weights']
    ncomponents = stick_weights.shape[1] + 1

    logprob = pd.DataFrame(data=np.zeros((len(class_samples), ncomponents)), index=class_samples.index,
                           columns=['Component ' + str(k) for k in range(ncomponents)])

    stick_running_product = 1.0
    cluster_weights_sum = 0.0
    for k in range(ncomponents):
        if k < (ncomponents - 1):
            cluster_weights = stick_weights[str(k)] * stick_running_product
            cluster_weights_sum += cluster_weights
            stick_running_product *= (1.0 - stick_weights[str(k)])
        else:
            cluster_weights = 1.0 - cluster_weights_sum

        clabel = 'Component ' + str(k)
        component_samples = class_samples[clabel]
        loglik = get_loglik_samples_one_class_single_component(component_samples, counts)
        logprob[clabel] = np.log(cluster_weights) + loglik

    logprob = logprob.apply(log_sum_of_logs, axis=1)

    return logprob


def get_loglik_samples_one_class_single_component(class_samples, counts):
    """
    Return logarithm of unnormalized class probability for one class for each MCMC sample.

    :param class_samples: MCMC samples for distribution corresponding to this class.
    :param counts: Feature vector for a single data point.
    :return:
    """
    counts = counts.values
    nfailure = class_samples['nfailure']
    beta_a = class_samples['beta_a']
    beta_b = class_samples['beta_b']
    alpha_cols = [c for c in class_samples.columns if 'alpha' in c]
    alpha = class_samples[alpha_cols]

    alpha_sum = np.sum(alpha, axis=1)

    total_counts = counts.sum() - 1
    gammaln_counts_per_bin = gammaln(counts + 1)
    gammaln_counts_minus_1 = gammaln(total_counts + 1)  # negative binomial parameter is wrt n - 1
    gammaln_counts = gammaln(counts.sum() + 1.0)

    loglik = gammaln(beta_a + beta_b) - \
             gammaln_counts_minus_1 - \
             gammaln(beta_a) - \
             gammaln(beta_b) - \
             gammaln(nfailure) + \
             gammaln(beta_a + nfailure) + \
             gammaln(total_counts + nfailure) + \
             gammaln(total_counts + beta_b) - \
             gammaln(total_counts + beta_a + beta_b + nfailure) + \
             gammaln_counts + \
             gammaln(alpha_sum) - \
             gammaln(total_counts + 1 + alpha_sum) + \
             np.sum(gammaln(counts + alpha) -
                    gammaln_counts_per_bin -
                    gammaln(alpha), axis=1)

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
        loglik_samples_this_class = get_loglik_samples_one_class_multi_component(samples[label], counts)
        # make sure we add in contribution from prior
        prob_samples[label] = loglik_samples_this_class + np.log(class_prior_prob[label])

    # renormalize so that max(unnormalized probability) = 1, improves numerical stability
    prob_samples = prob_samples.apply(lambda x: x - prob_samples.max(axis=1))
    prob_samples = np.exp(prob_samples)

    # normalize so probabilities sum to one for each MCMC sample
    prob_samples = prob_samples.apply(lambda x: x / prob_samples.sum(axis=1))

    return prob_samples


def rhat_class_predictions(prob_samples):
    prob_samples = logit(prob_samples)
    # first split the chains in half
    nsamples = len(prob_samples.index.get_level_values(1).unique())
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


def get_prob_samples_helper_(args):
    mcmc_samples, x, class_prior, compute_rhat = args
    print x.name
    prob_samples = get_prob_samples_all_classes(mcmc_samples, x, class_prior)

    if compute_rhat:
        this_rhat = rhat_class_predictions(prob_samples)
    else:
        this_rhat = None

    prob_samples = prob_samples.mean(axis=0)

    return prob_samples, this_rhat


def classify(samples, counts, prior_counts, compute_rhat=False, n_jobs=1):
    fcols = ['feat_' + str(k+1) for k in range(93)]
    counts = counts[fcols]
    class_prior_prob = pd.DataFrame(np.random.dirichlet(prior_counts + 1, size=samples.shape[0]), index=samples.index,
                                    columns=prior_counts.index)

    args_list = [(samples, counts.loc[idx], class_prior_prob, compute_rhat) for idx in counts.index]

    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()
    if n_jobs > 1:
        pool = multiprocessing.Pool(n_jobs, maxtasksperchild=10)
        output_list = pool.map(get_prob_samples_helper_, args_list)
    else:
        output_list = map(get_prob_samples_helper_, args_list)

    class_probs_list = [p for p, r in output_list]
    class_probs = pd.concat(class_probs_list, axis=1, ignore_index=True).T
    class_probs.index = counts.index

    if compute_rhat:
        rhat_list = [r for p, r in output_list]
        rhat = pd.concat(rhat_list, axis=1, ignore_index=True).T
        rhat.index = counts.index

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