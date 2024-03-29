__author__ = 'brandonkelly'

import numpy as np
from scipy.special import gammaln
from bck_mcmc.parameter import Parameter
from scipy import linalg
from sklearn.cluster import KMeans


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
        super(MixLogNegBinPars, self).__init__(label, track)
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
        prior_mu.children.append(self)
        prior_covar.children.append(self)

    def initialize(self):
        if self.prior_mu is None or self.prior_covar is None or self.components is None:
            raise ValueError("Must initialize the prior and component objects.")

        beta_a = np.random.uniform(2.0, 3.0)
        beta_b = np.random.uniform(1.0, 3.0)

        nk = np.sum(self.components.value == self.component_label)
        if nk > 1:
            # estimate nfailure from mean counts
            mean_counts = np.mean(self.counts[self.components.value == self.component_label])
            nfailure = mean_counts * beta_b / (beta_a - 1.0) * (1.0 + np.random.uniform(-0.1, 0.1))
        else:
            nfailure = np.random.uniform(5, 100)

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

        vcentered = value - self.prior_mu.value
        logpost = -0.5 * vcentered.dot(linalg.inv(self.prior_covar.value).dot(vcentered))  # the prior

        if ndata > 0:
            # likelihood contribution
            logpost += ndata * (gammaln(beta_a + beta_b) - gammaln(beta_a) - gammaln(beta_b) +
                                gammaln(beta_a + nfailure) - gammaln(nfailure))

            # only compute log gamma function for unique values of self.counts, since expensive
            gammaln_counts_nfailure_unique = gammaln(self.unique_counts + nfailure)
            gammaln_counts_b_unique = gammaln(self.unique_counts + beta_b)
            gammaln_counts_abn_unique = gammaln(self.unique_counts + beta_a + beta_b + nfailure)

            logpost += np.sum(gammaln_counts_nfailure_unique[unique_idx] + gammaln_counts_b_unique[unique_idx] -
                              gammaln_counts_abn_unique[unique_idx])

        return logpost


def alpha_transform(log_alpha):
    alpha = np.exp(log_alpha)
    alpha_0 = np.log(alpha.sum())
    y = np.log(alpha[:-1] / alpha[-1])

    return np.concatenate(([alpha_0], y))


def alpha_inverse_transform(alpha_inverse):
    alpha_sum = np.exp(alpha_inverse[0])
    y = alpha_inverse[1:]
    alpha = alpha_sum * np.exp(y) / (1.0 + np.sum(np.exp(y)))
    alpha = np.append(alpha, alpha_sum / (1.0 + np.sum(np.exp(y))))

    return alpha


class LogBinProbsAlpha(Parameter):
    def __init__(self, counts_per_bin, label, component_label, track=True):
        super(LogBinProbsAlpha, self).__init__(label, track)
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
        nk = np.sum(component_idx)
        if nk > 1:
            counts_per_bin_k = self.counts_per_bin[component_idx] + 1.0
            bin_fracs_k = counts_per_bin_k / counts_per_bin_k.sum(axis=1)[:, np.newaxis].astype(float)
            avg_bin_frac = bin_fracs_k.mean(axis=0)
            var_bin_frac = bin_fracs_k.var(axis=0)

            concentration = np.median(avg_bin_frac * (1.0 - avg_bin_frac) / var_bin_frac - 1.0)
            concentration = max(concentration, 1.0)
            alphas = concentration * avg_bin_frac
        else:
            alphas = np.ones(self.counts_per_bin.shape[1])

        self.value = alpha_transform(np.log(alphas))

    def logdensity(self, value):
        # transform values
        alphas = alpha_inverse_transform(value)
        concentration = alphas.sum()
        # y = np.log(alphas[:-1] / alphas[-1])

        # values corresponding to the subset of data associated with this mixture component
        component_idx = self.components.value == self.component_label
        counts_per_bin_k = self.counts_per_bin[component_idx]
        unique_idx_k = self.unique_idx[component_idx]
        ndata_k = np.sum(component_idx)

        # the prior

        # jacobian = -np.sum(value[:-1]) + np.log(1.0 + np.sum(alphas[:-1] / alphas[-1])) - value_transformed[0]
        jacobian = 0.0
        # vcentered = value_transformed - self.prior_mu.value
        vcentered = value - self.prior_mu.value
        logprior = jacobian + np.sum(-0.5 * vcentered ** 2 / self.prior_var.value)

        if ndata_k > 0:
            # the likelihood

            gammaln_bin_counts_alpha = gammaln(counts_per_bin_k + alphas)
            gammaln_counts_conc = gammaln(self.unique_counts + concentration)

            # sum loglik over bins for each data point
            loglik = self.gammaln_counts[component_idx].sum() + \
                     ndata_k * gammaln(concentration) - \
                     np.sum(gammaln_counts_conc[unique_idx_k]) + \
                     np.sum(gammaln_bin_counts_alpha) - \
                     np.sum(self.gammaln_counts_per_bin[component_idx]) - \
                     ndata_k * np.sum(gammaln(alphas))

        else:
            loglik = 0.0

        logpost = loglik + logprior
        return logpost


class MixtureComponents(Parameter):
    def __init__(self, label, counts_per_bin, track=True):
        super(MixtureComponents, self).__init__(label, track)
        self.stick_weights = None
        self.negbin_pars = dict()
        self.alphas = dict()
        self.ncomponents = 0
        self.counts_per_bin = counts_per_bin
        self.ndata = counts_per_bin.shape[0]

        # precompute quantities
        self.gammaln_counts_per_bin = gammaln(counts_per_bin + 1)
        self.gammaln_counts_minus_1 = gammaln(counts_per_bin.sum(axis=1))  # negative binomial parameter is wrt n - 1
        self.gammaln_counts = gammaln(counts_per_bin.sum(axis=1) + 1.0)

    def add_component(self, negbin, alpha, component_label):
        self.negbin_pars[component_label] = negbin
        self.negbin_pars[component_label].components = self
        self.alphas[component_label] = alpha
        self.alphas[component_label].components = self
        self.ncomponents = len(self.negbin_pars)

    def initialize(self):
        if self.ncomponents == 0:
            raise ValueError("Component parameter dictionary is empty.")
        if self.stick_weights is None:
            raise ValueError(self.label + " does not know about a stick weight parameter.")

        total_counts = self.counts_per_bin.sum(axis=1)
        counts_per_bin = self.counts_per_bin + 1.0
        bin_fracs = counts_per_bin / counts_per_bin.sum(axis=1)[:, np.newaxis].astype(float)
        def logit(x):
            return np.log(x / (1.0 - x))
        logit_bin_fracs = logit(bin_fracs)
        X = np.column_stack((np.log(total_counts), logit_bin_fracs))
        cluster_labels = KMeans(n_clusters=self.ncomponents).fit_predict(X)
        # order the clusters by decreasing counts
        cluster_counts = np.bincount(cluster_labels, minlength=self.ncomponents)
        sorted_idx = np.argsort(cluster_counts)[::-1]
        self.value = np.zeros_like(cluster_labels)
        for i in range(self.ncomponents):
            self.value[cluster_labels == sorted_idx[i]] = i

    def initialize2(self):
        if self.ncomponents == 0:
            raise ValueError("Component parameter dictionary is empty.")
        if self.stick_weights is None:
            raise ValueError(self.label + " does not know about a stick weight parameter.")

        # initialize size of clusters by drawing the stick breaking parameter from its prior, and then performing a
        # multinomial draw

        cluster_weights = np.zeros(self.ncomponents)
        stick_weights = np.random.beta(1.0, 1.0, self.ncomponents - 1)
        cluster_weights[0] = stick_weights[0]
        for k in range(1, self.ncomponents - 1):
            cluster_weights[k] = stick_weights[k] * np.prod(1.0 - stick_weights[:k])
        cluster_weights[-1] = 1.0 - np.sum(cluster_weights[:-1])

        assert cluster_weights.sum() == 1

        # randomly assign data to clusters
        self.value = np.random.choice(self.ncomponents, size=self.ndata, p=cluster_weights)

    def random_draw(self):

        counts = self.counts_per_bin.sum(axis=1) - 1

        # first compute part of posterior that can be vectorized
        logpost = np.empty((self.ndata, self.ncomponents))

        # get the cluster weights
        cluster_weights = np.zeros(self.ncomponents)
        cluster_weights[0] = self.stick_weights.value[0]
        for k in range(1, self.ncomponents - 1):
            cluster_weights[k] = self.stick_weights.value[k] * np.prod(1.0 - self.stick_weights.value[:k])
        cluster_weights[-1] = 1.0 - np.sum(cluster_weights[:-1])

        for k in range(self.ncomponents):
            beta_a = np.exp(self.negbin_pars[k].value[1])
            beta_b = np.exp(self.negbin_pars[k].value[2])
            nfailure = np.exp(self.negbin_pars[k].value[0])
            alpha = alpha_inverse_transform(self.alphas[k].value)
            # alpha = np.exp(self.alphas[k].value)
            alpha_sum = np.sum(alpha)

            logpost[:, k] = gammaln(beta_a + beta_b) - \
                            self.gammaln_counts_minus_1 - \
                            gammaln(beta_a) - \
                            gammaln(beta_b) - \
                            gammaln(nfailure) + \
                            gammaln(beta_a + nfailure) + \
                            gammaln(counts + nfailure) + \
                            gammaln(counts + beta_b) - \
                            gammaln(counts + beta_a + beta_b + nfailure) + \
                            self.gammaln_counts + \
                            gammaln(alpha_sum) - \
                            gammaln(counts + 1 + alpha_sum) + \
                            np.sum(gammaln(self.counts_per_bin + alpha) -
                                   self.gammaln_counts_per_bin -
                                   gammaln(alpha), axis=1) + \
                            np.log(cluster_weights[k])

        cluster_prob = np.exp(logpost - logpost.max(axis=1)[:, np.newaxis])
        cluster_prob /= cluster_prob.sum(axis=1)[:, np.newaxis]

        # update the component labels
        new_component_labels = np.empty_like(self.value)
        for i in range(self.ndata):
            new_component_labels[i] = np.random.choice(self.ncomponents, p=cluster_prob[i])

        return new_component_labels


class StickWeight(Parameter):
    def __init__(self, label, track=True):
        super(StickWeight, self).__init__(label, track)
        self.components = None
        self.concentration = None

    def initialize(self):
        if self.components is None or self.concentration is None:
            raise ValueError(self.label + " does not know about the component labels or the DP concentration.")
        self.value = self.random_draw()

    def random_draw(self):
        new_weight = np.empty(self.components.ncomponents - 1)
        n_k = np.empty(self.components.ncomponents)
        for k in range(self.components.ncomponents):
            n_k[k] = np.sum(self.components.value == k)
        for k in range(self.components.ncomponents - 1):
            new_weight[k] = np.random.beta(1 + n_k[k], self.concentration.value + np.sum(n_k[k+1:]))

        # correction for numerical stability
        new_weight[new_weight > 0.999] = 0.999

        return new_weight


class DPconcentration(Parameter):
    def __init__(self, label, prior_shape, prior_scale, track=True):
        super(DPconcentration, self).__init__(label, track)
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale
        self.stick_weights = None

    def initialize(self):
        if self.stick_weights is None:
            raise ValueError(self.label + ' does not know about a stick weight parameter.')
        # just draw from prior
        self.value = np.random.gamma(self.prior_shape, 1.0 / self.prior_scale)

    def random_draw(self):
        ncomponents = len(self.stick_weights.value) + 1.0
        shape = ncomponents - 1 + self.prior_shape
        scale = self.prior_scale - np.sum(np.log(1.0 - self.stick_weights.value))

        return np.random.gamma(shape, 1.0 / scale)


class PriorMu(Parameter):
    def __init__(self, label, prior_mean, prior_var, track=True, transform=None):
        super(PriorMu, self).__init__(label, track)
        self.children = []
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.child_var = None
        self.transform = transform

    def initialize(self):
        if not self.children:
            raise ValueError("Must connect children.")
        if self.child_var is None:
            raise ValueError("Unknown child variance parameter.")

        data_sum = 0.0
        for child in self.children:
            if self.transform is None:
                data_sum += child.value
            else:
                data_sum += self.transform(child.value)

        self.value = data_sum / len(self.children)

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
            if self.transform is None:
                data_sum += child.value
            else:
                data_sum += self.transform(child.value)

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
            post_var = np.diag(post_var)

        return np.random.multivariate_normal(post_mean, post_var)


class PriorVar(Parameter):
    def __init__(self, label, prior_dof, prior_ssqr, track=True, transform=None):
        super(PriorVar, self).__init__(label, track)
        self.children = []
        self.prior_dof = prior_dof
        self.prior_ssqr = prior_ssqr
        self.child_mean = None
        self.transform = transform

    def initialize(self):
        if not self.children:
            raise ValueError("Must connect children.")
        if self.child_mean is None:
            raise ValueError("Unknown child mean parameter.")

        self.value = self.random_draw()

    def random_draw(self):

        nchildren = len(self.children)
        data_ssqr = 0.0
        for child in self.children:
            if self.transform is None:
                child_value = child.value
            else:
                child_value = self.transform(child.value)
            data_ssqr += (child_value - self.child_mean.value) ** 2

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
            data_scale += np.outer(child.value - self.child_mean.value, child.value - self.child_mean.value)

        post_scale = data_scale + self.prior_scale
        post_precision = linalg.inv(post_scale)
        x = np.random.multivariate_normal(np.zeros(post_precision.shape[0]), post_precision, post_dof)
        A_inv = 0.0
        for j in range(post_dof):
            A_inv += np.outer(x[j], x[j])

        A = linalg.inv(A_inv)
        return A