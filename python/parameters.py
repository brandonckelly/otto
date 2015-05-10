__author__ = 'brandonkelly'

import numpy as np
from scipy.special import gammaln
from bck_mcmc.parameter import Parameter
from scipy import linalg


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
        fractions = (1.0 + self.counts_per_bin) / (1.0 + self.counts[:, np.newaxis].astype(float))

        fraction = np.mean(fractions, axis=0)
        fraction /= fraction.sum()

        concentration = fraction * (1.0 - fraction) / fractions.var(axis=0) - 1.0
        concentration[concentration <= 0] = 1.0
        chat = np.median(concentration)
        if chat > 1e3:
            chat = 1e3
        elif chat < 1.0:
            chat = 1.0
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
        shape_pars = (self.counts_per_bin.sum(axis=0) + 1.0) / 2.0
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


# ######### parameters for mixture ##########


class MixLogNegBinPars(Parameter):
    def __init__(self, counts, label, component_label, track=True):
        super(LogNegBinCounts, self).__init__(label, track)
        self.counts = counts - 1  # data
        self.prior_mu = None
        self.prior_covar = None
        self.components = None
        self.component_label = component_label

        # precompute quantities
        self.gammaln_counts = gammaln(self.counts + 1.0)
        unique_counts, unique_idx = np.unique(self.counts, return_inverse=True)
        self.unique_idx = unique_idx
        self.unique_counts = unique_counts

    def connect_prior(self, prior_mu, prior_covar):
        # can't have more than one parent
        if self.prior_mu is not None:
            self.prior_mu.children.remove(self)
        if self.prior_covar is not None:
            self.prior_covar.children.remove(self)

        self.prior_mu = prior_mu
        self.prior_covar = prior_covar
        prior_mu.add_child(self)
        prior_covar.add_child(self)

    def initialize(self):
        if self.prior_mu is None or self.prior_covar is None or self.components is None:
            raise ValueError("Must initialize the prior and component objects.")

        beta_a = np.random.uniform(1.0, 3.0)
        beta_b = np.random.uniform(1.0, 3.0)

        # estimate nfailure from mean
        mean_counts = np.mean(self.counts[self.components.value == self.component_label])
        nfailure = mean_counts * beta_b / (beta_a - 1.0) * (1.0 + np.random.uniform(-0.1, 0.1))

        self.value = np.log([nfailure, beta_a, beta_b])

    def logdensity(self, value):

        nfailure = np.exp(value[0])
        beta_a = np.exp(value[1])
        beta_b = np.exp(value[2])

        # value corresponding to the subset of data associated with this mixture component
        component_idx = self.components.value == self.component_label
        component_counts = self.counts[component_idx]
        unique_idx = self.unique_idx[component_idx]
        ndata = len(component_counts)
        total_counts = np.sum(component_counts)

        vcentered = value - self.prior_mu.value
        logpost = -0.5 * vcentered.dot(linalg.inv(self.prior_covar.value).dot(vcentered))  # the prior

        # likelihood contribution
        logpost += gammaln(beta_a + total_counts) + \
                   gammaln(beta_b + ndata * nfailure) - \
                   gammaln(beta_a + beta_b + total_counts + ndata * nfailure)

        # only compute log gamma function for unique values of self.counts, since expensive
        gammaln_counts_value_unique = gammaln(self.unique_counts + nfailure)
        logpost += np.sum(gammaln_counts_value_unique[unique_idx] - self.gammaln_counts[component_idx]) - \
                   ndata * gammaln(nfailure)

        return logpost


class LogBinProbsAlpha(Parameter):
    def __init__(self, counts_per_bin, label, component_label, track=True):
        super(LogBinProbsGamma, self).__init__(label, track)
        self.prior_mu = None
        self.prior_var = None
        self.counts_per_bin = counts_per_bin
        self.components = None
        self.component_label = component_label

        # precompute quantities
        self.counts = np.sum(counts_per_bin, axis=1)
        self.gammaln_counts = gammaln(self.counts + 1.0)
        self.gammaln_counts_per_bin = gammaln(self.counts_per_bin + 1.0)
        unique_counts, unique_idx = np.unique(self.counts, return_inverse=True)
        self.unique_counts = unique_counts
        self.unique_idx = unique_idx

    def connect_prior(self, prior_mu, prior_var):
        # can't have more than one parent
        if self.prior_mu is not None:
            self.prior_mu.children.remove(self)
        if self.prior_var is not None:
            self.prior_var.children.remove(self)

        self.prior_mu = prior_mu
        self.prior_var = prior_var
        self.prior_mu.child_var = prior_var
        self.prior_var.child_mu = prior_mu
        prior_mu.children.append(self)
        prior_var.children.append(self)

    def initialize(self):
        if self.prior_mu is None or self.prior_var is None or self.components is None:
            raise ValueError("Must initialize prior and component objects.")

        component_idx = self.components.value == self.component_label
        counts_per_bin_k = self.counts_per_bin[component_idx]
        bin_fracs_k = counts_per_bin_k / counts_per_bin_k.sum(axis=1)[:, np.newaxis].astype(float)
        avg_bin_frac = bin_fracs_k.mean(axis=0)
        var_bin_frac = bin_fracs_k.var(axis=0)

        concentration = np.median(avg_bin_frac * (1.0 - avg_bin_frac) / var_bin_frac - 1.0)
        concentration = max(concentration, 1.0)
        alphas = concentration * avg_bin_frac

        self.value = np.log(alphas)

    def logdensity(self, value):
        # transform values
        alphas = np.exp(value)
        concentration = alphas.sum()
        y = np.log(alphas[:-1] / alphas[-1])

        # values corresponding to the subset of data associated with this mixture component
        component_idx = self.components.value == self.component_label
        counts_per_bin_k = self.counts_per_bin[component_idx]
        unique_idx_k = self.unique_idx[component_idx]
        ndata_k = np.sum(component_idx)

        # the prior

        value_transformed = np.concatenate((np.log([concentration]), y))
        jacobian = -value[:-1] + np.log(1.0 + np.sum(alphas[:-1] / alphas[-1])) - value_transformed[0]
        vcentered = value_transformed - self.prior_mu.value
        logprior = jacobian + np.sum(-0.5 * vcentered ** 2 / self.prior_var)

        # ### the likelihood

        gammaln_bin_counts_alpha = gammaln(counts_per_bin_k + alphas)
        gammaln_counts_conc = gammaln(self.unique_counts + concentration)

        # sum loglik over bins for each data point
        loglik = self.gammaln_counts[component_idx].sum() + \
                 ndata_k * gammaln(concentration) - \
                 np.sum(gammaln_counts_conc[unique_idx_k]) + \
                 np.sum(gammaln_bin_counts_alpha) - \
                 np.sum(self.gammaln_counts_per_bin[component_idx]) - \
                 ndata_k * np.sum(gammaln(alphas))

        logpost = loglik + logprior
        return logpost


class MixtureComponents(Parameter):
    pass


class MixtureWeight(Parameter):
    pass


class PriorMu(Parameter):
    def __init__(self, label, prior_mean, prior_var, track=True):
        super(PriorVar, self).__init__(label, track)
        self.children = []
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.child_var = None

    def initialize(self):
        if not self.children:
            raise ValueError("Must connect children.")
        if self.child_var is None:
            raise ValueError("Unknown child variance parameter.")

        self.value = self.random_draw()

    def random_draw(self):
        if self.prior_var.ndim == 2:
            # covariance matrix
            post_precision = linalg.inv(self.prior_var)
            post_mean = post_precision.dot(self.prior_mean)
        else:
            # diagonal matrix
            post_precision = 1.0 / self.prior_var
            post_mean = post_precision * self.prior_mean

        data_sum = 0.0
        for child in self.children:
            data_sum += child.value

        if self.child_var.value.ndim == 2:
            # covariance matrix
            data_precision = linalg.inv(self.child_var.value)
            post_mean += data_precision.dot(data_sum)
        else:
            # diagonal covariance matrix
            data_precision = 1.0 / self.child_var.value
            post_mean += data_precision * data_sum

        post_precision += len(self.children) * data_precision

        if post_precision.ndim == 2:
            post_var = linalg.inv(post_precision)
            post_mean = post_var.dot(post_mean)
        else:
            post_var = 1.0 / post_precision
            post_mean *= post_var

        return np.random.multivariate_normal(post_mean, post_var)


class PriorVar(Parameter):
    def __init__(self, label, prior_dof, prior_ssqr, track=True):
        super(PriorVar, self).__init__(label, track)
        self.children = []
        self.prior_dof = prior_dof
        self.prior_ssqr = prior_ssqr
        self.child_mean = None

    def initialize(self):
        if not self.children:
            raise ValueError("Must connect children.")
        if self.child_mean is None:
            raise ValueError("Unknown child variance parameter.")

        self.value = self.random_draw()

    def random_draw(self):

        nchildren = len(self.children)
        data_ssqr = 0.0
        for child in self.children:
            data_ssqr += (child.value - self.child_mean.value) ** 2

        post_dof = self.prior_dof + nchildren
        post_ssqr = (self.prior_dof * self.prior_ssqr + data_ssqr) / post_dof

        return post_ssqr * post_dof / np.random.chisquare(post_dof)


class PriorCovar(Parameter):
    def __init__(self, label, prior_dof, prior_scale, track=True):
        super(PriorCovar, self).__init__(label, track)
        self.children = []
        self.prior_dof = prior_dof
        self.prior_scale = prior_scale
        self.child_mean = None

    def initialize(self):
        if not self.children:
            raise ValueError("Must connect children.")
        if self.child_mean is None:
            raise ValueError("Unknown child variance parameter.")

        self.value = self.random_draw()

    def random_draw(self):
        nchildren = len(self.children)
        post_dof = nchildren + self.prior_dof

        data_scale = 0.0
        for child in self.children:
            data_scale += np.outer(child.value - self.child_mean, child.value - self.child_mean)

        post_scale = data_scale + self.prior_scale
        post_precision = linalg.inv(post_scale)
        x = np.random.multivariate_normal(np.zeros(post_precision.shape[0]), post_precision, post_dof)
        A_inv = 0.0
        for j in range(post_dof):
            A_inv += np.outer(x[j], x[j])

        A = linalg.inv(A_inv)
        return A