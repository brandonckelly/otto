__author__ = 'brandonkelly'

import numpy as np
from scipy.special import gammaln
from bck_mcmc.parameter import Parameter


class LogNegBinCounts(Parameter):

    def __init__(self, counts, label, track=True, prior_a=1.0, prior_b=1.0, prior_mu=4.6, prior_sigma=2.0):
        super(LogNegBinCounts, self).__init__(label, track)
        self.counts = counts - 1  # data
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        # precompute quantities
        self.total_counts = np.sum(counts)
        self.ndata = len(counts)
        self.gammaln_a_total_counts = gammaln(self.prior_a + self.total_counts)
        self.gammaln_counts = gammaln(self.counts + 1.0)
        unique_counts, unique_idx = np.unique(self.counts, return_inverse=True)
        self.unique_idx = unique_idx
        self.unique_counts = unique_counts

    def initialize(self):
        # compute marginal posterior on a grid an draw from the pdf
        rgrid = np.linspace(1.0, 100.0, 10000)
        logpost_grid = np.zeros(len(rgrid))
        for i in range(len(rgrid)):
            logpost_grid[i] = self.logdensity(np.log(rgrid[i]))

        # refine the grid to focus on regions with high probability
        max_idx = np.argmax(logpost_grid)
        thresh = logpost_grid[max_idx] - np.log(200)
        lowest_idx = np.sum(logpost_grid[:max_idx] < thresh) + 1
        new_low = rgrid[lowest_idx]
        highest_idx = np.sum(logpost_grid[max_idx:] >= thresh) + max_idx - 1
        new_high = rgrid[highest_idx]
        rgrid = np.linspace(new_low, new_high, 1000)
        logpost_grid = np.zeros(len(rgrid))
        for i in range(len(rgrid)):
            logpost_grid[i] = self.logdensity(np.log(rgrid[i]))

        # compute the pdf and draw from it
        r_marginal = np.exp(logpost_grid - logpost_grid.max())
        r_marginal /= r_marginal.sum()

        ivalue = np.random.choice(rgrid, p=r_marginal)

        self.value = np.log(ivalue)

    def logdensity(self, value):

        nfailure = np.exp(value)

        logpost = -0.5 * (value - self.prior_mu) ** 2 / self.prior_sigma ** 2  # the prior
        logpost += self.gammaln_a_total_counts + \
                  gammaln(self.prior_b + self.ndata * nfailure) - \
                  gammaln(self.prior_a + self.prior_b + self.total_counts + self.ndata * nfailure)
        # only compute log gamma function for unique values of self.counts, since expensive
        gammaln_counts_value_unique = gammaln(self.unique_counts + nfailure)
        logpost += np.sum(gammaln_counts_value_unique[self.unique_idx] - self.gammaln_counts) - \
                   self.ndata * gammaln(nfailure)

        return logpost


class LogConcentration(Parameter):

    def __init__(self, counts_per_bin, label, track=True, prior_mean=0.0, prior_sigma=10.0):
        super(LogConcentration, self).__init__(label, track)
        self.counts_per_bin = counts_per_bin
        self.log_gamma = None
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma

        # precompute quantities
        self.counts = np.sum(counts_per_bin, axis=1)
        self.ndata = len(self.counts)
        self.gammaln_counts = gammaln(self.counts + 1.0)
        self.gammaln_counts_per_bin = gammaln(self.counts_per_bin + 1.0)
        unique_counts, unique_idx = np.unique(self.counts, return_inverse=True)
        self.unique_counts = unique_counts
        self.unique_idx = unique_idx

    def initialize(self):
        if self.log_gamma is None:
            raise ValueError("Associated LogBinProbsGamma instance is not initialized.")

        # estimate concentration by matching moments
        fractions = self.counts_per_bin / self.counts[:, np.newaxis].astype(float)

        fraction = np.mean(fractions, axis=0)
        fraction /= fraction.sum()

        concentration = fraction * (1.0 - fraction) / fractions.var(axis=0) - 1.0
        concentration[concentration <= 0] = 1.0
        chat = np.median(concentration)

        cguess = np.random.lognormal(np.log(chat), 0.05)  # perturb the estimate by 5%
        self.value = cguess

    def connect_log_gamma(self, bprobs):
        self.log_gamma = bprobs
        bprobs.concentration = self

    def logdensity(self, value):

        # transform values
        concentration = np.exp(value)
        bin_probs_values = np.exp(self.log_gamma.value)
        bin_probs_values /= np.sum(bin_probs_values)

        logprior = -0.5 * (value - self.prior_mean) ** 2 / self.prior_sigma ** 2

        # compute log-gamma function only for unique values for efficiency

        bcounts_sum = self.counts_per_bin + concentration * bin_probs_values
        uniq_bcsum, u_idx = np.unique(bcounts_sum, return_inverse=True)
        gammaln_bin_counts_conc = gammaln(uniq_bcsum)[u_idx].reshape(bcounts_sum.shape)
        gammaln_counts_conc = gammaln(self.unique_counts + concentration)

        # sum loglik over bins for each data point
        loglik = np.sum(self.gammaln_counts) + \
                 self.ndata * gammaln(concentration) - \
                 np.sum(gammaln_counts_conc[self.unique_idx]) + \
                 np.sum(gammaln_bin_counts_conc) - \
                 np.sum(self.gammaln_counts_per_bin) - \
                 self.ndata * np.sum(gammaln(concentration * bin_probs_values))

        logpost = loglik + logprior
        return logpost


# TODO: Create Scalar Version
class LogBinProbsGamma(Parameter):

    def __init__(self, counts_per_bin, label, track=True, prior_alphas=1.0):
        super(LogBinProbsGamma, self).__init__(label, track)
        self.prior_alphas = prior_alphas
        self.counts_per_bin = counts_per_bin
        self.concentration = None

        # precompute quantities
        self.counts = np.sum(counts_per_bin, axis=1)
        self.ndata = len(self.counts)
        self.gammaln_prior_alphas = gammaln(prior_alphas)
        self.gammaln_counts = gammaln(self.counts + 1.0)
        self.gammaln_counts_per_bin = gammaln(self.counts_per_bin + 1.0)
        unique_counts, unique_idx = np.unique(self.counts, return_inverse=True)
        self.unique_counts = unique_counts
        self.unique_idx = unique_idx

    def connect_concentration(self, conc):
        self.concentration = conc
        conc.log_gamma = self

    def initialize(self):
        if self.concentration is None:
            raise ValueError("Associated LogConcentration instance is not initialized.")

        # bin_prob_guess = np.random.dirichlet(self.counts_per_bin.sum(axis=0) / 10.0)
        shape_pars = self.counts_per_bin.sum(axis=0) / 10.0
        gammas = np.random.gamma(shape_pars)
        self.value = np.log(gammas)

    def logdensity(self, value):

        # transform values
        gammas = np.exp(value)
        bin_probs = gammas / np.sum(gammas)
        concentration = np.exp(self.concentration.value)

        logprior = value + (self.prior_alphas - 1.0) * value - gammas - self.gammaln_prior_alphas
        logprior = np.sum(logprior)

        # compute log-gamma function only for unique values for efficiency
        bcounts_sum = self.counts_per_bin + concentration * bin_probs
        uniq_bcsum, u_idx = np.unique(bcounts_sum, return_inverse=True)
        gammaln_bin_counts_conc = gammaln(uniq_bcsum)[u_idx].reshape(bcounts_sum.shape)

        gammaln_counts_conc = gammaln(self.unique_counts + concentration)

        # sum loglik over bins for each data point
        loglik = np.sum(self.gammaln_counts) + \
                 self.ndata * gammaln(concentration) - \
                 np.sum(gammaln_counts_conc[self.unique_idx]) + \
                 np.sum(gammaln_bin_counts_conc) - \
                 np.sum(self.gammaln_counts_per_bin) - \
                 self.ndata * np.sum(gammaln(concentration * bin_probs))

        logpost = loglik + logprior
        return logpost

