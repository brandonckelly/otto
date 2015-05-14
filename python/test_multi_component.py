__author__ = 'brandonkelly'

__author__ = 'brandonkelly'

import unittest
import numpy as np
from bck_mcmc.sampler import Sampler
from bck_mcmc.steps import RobustAdaptiveMetro
from parameters import DPconcentration, StickWeight, MixtureComponents, MixLogNegBinPars, LogBinProbsAlpha, \
    PriorCovar, PriorMu, PriorVar
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from run_multi_component_sampler import alpha_transform, alpha_inverse_transform
from posterior_predictive_check import negbin_sample


def bnb_sample(r, a, b, n):
    p = np.random.beta(a, b, n)
    m = negbin_sample(r, p)
    return m


class TestParameters(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)


        self.ndata = 1000
        self.nbins = 10
        self.ncomponents = 100
        self.bin_counts = np.zeros((self.ndata, self.nbins), dtype=np.int)

        self.dp_concentration = 1.0
        self.stick_weights = np.random.beta(1.0, self.dp_concentration, self.ncomponents)
        self.cluster_weights = np.zeros(self.ncomponents)
        self.cluster_weights[0] = self.stick_weights[0]
        for k in range(1, self.ncomponents - 1):
            self.cluster_weights[k] = self.stick_weights[k] * np.prod(1.0 - self.stick_weights[:k])
        self.cluster_weights[-1] = 1.0 - np.sum(self.cluster_weights[:-1])

        self.component_labels = np.random.choice(self.ncomponents, size=self.ndata, p=self.cluster_weights)

        self.negbin_mu = np.array([np.log(40), np.log(4), np.log(4)])
        self.negbin_covar = np.array([[0.25 ** 2, -0.01, 0.005],
                                      [-0.01, 0.1 ** 2, 0.0],
                                      [0.005, 0.0, 0.1 ** 2]])

        negbin_pars = np.random.multivariate_normal(self.negbin_mu, self.negbin_covar, self.ncomponents)
        self.nfailures = np.exp(negbin_pars[:, 0])
        self.beta_a = np.exp(negbin_pars[:, 1])
        self.beta_b = np.exp(negbin_pars[:, 2])

        self.alpha_inverse_mu = np.array([np.log(5)] + [0] * (self.nbins - 1))
        self.alpha_inverse_var = np.ones_like(self.alpha_inverse_mu) * 0.1 ** 2

        self.alpha = np.zeros((self.nbins, self.ncomponents))
        self.alpha_inverse = np.zeros_like(self.alpha)
        for k in range(self.ncomponents):
            # on the log scale
            alpha_inverse = self.alpha_inverse_mu + \
                            np.sqrt(self.alpha_inverse_var) * np.random.standard_normal(self.nbins)
            self.alpha[:, k] = alpha_inverse_transform(alpha_inverse)
            self.alpha_inverse[:, k] = alpha_inverse

        for k in range(self.ncomponents):
            k_idx = np.where(self.component_labels == k)[0]
            nk = len(k_idx)
            if nk > 0:
                total_counts = bnb_sample(self.nfailures[k], self.beta_a[k], self.beta_b[k], nk)
                for i in range(nk):
                    probs = np.random.dirichlet(self.alpha[:, k])
                    if nk == 1:
                        self.bin_counts[k_idx[i]] = np.random.multinomial(total_counts, probs)
                    else:
                        self.bin_counts[k_idx[i]] = np.random.multinomial(total_counts[i], probs)

        self.niter = 20000
        self.nburn = 2500

    def test_dp_concentration(self):
        dp_conc = DPconcentration('dp-alpha', 1.0, 1.0)
        stick_weights = StickWeight('stick')
        stick_weights.value = self.stick_weights
        dp_conc.stick_weights = stick_weights
        dp_conc.initialize()
        dp_draws = np.empty(self.niter)
        for i in range(self.niter):
            dp_draws[i] = dp_conc.random_draw()

        dp_low = np.percentile(dp_draws, 2.5)
        dp_high = np.percentile(dp_draws, 97.5)

        self.assertLess(dp_low, self.dp_concentration)
        self.assertGreater(dp_high, self.dp_concentration)

        plt.hist(dp_draws, bins=100)
        plt.vlines(self.dp_concentration, plt.ylim()[0], plt.ylim()[1])
        plt.xlabel('DP Concentration')
        plt.show()

    def test_stick_weights(self):
        stick_weights = StickWeight('stick')
        dp_conc = DPconcentration('dp-alpha', 1.0, 1.0)
        labels = MixtureComponents('labels', self.bin_counts)

        dp_conc.value = self.dp_concentration
        labels.value = self.component_labels
        labels.ncomponents = self.ncomponents

        stick_weights.components = labels
        stick_weights.concentration = dp_conc

        stick_weights.initialize()
        stick_draws = np.empty((self.niter, self.ncomponents - 1))
        for i in range(self.niter):
            stick_draws[i] = stick_weights.random_draw()

        s_low = np.percentile(stick_draws, 1.0, axis=0)
        s_high = np.percentile(stick_draws, 99.0, axis=0)

        for k in range(self.ncomponents - 1):
            self.assertLess(s_low[k], self.stick_weights[k])
            self.assertGreater(s_high[k], self.stick_weights[k])
            plt.hist(stick_draws[:, k], bins=100)
            plt.vlines(self.stick_weights[k], plt.ylim()[0], plt.ylim()[1])
            plt.xlabel('Stick Weight ' + str(k))
            plt.show()

    def test_prior_mu(self):

        # first test prior mean without a transform
        mu = PriorMu('prior-mu', np.zeros(3), 10.0 * np.identity(3))
        sigsqr = PriorCovar('prior-covar', 3 * np.ones(3), np.identity(3))
        sigsqr.value = self.negbin_covar
        mu.child_var = sigsqr

        for k in range(self.ncomponents):
            negbin_k = MixLogNegBinPars(self.bin_counts.sum(axis=1), 'log-negbin-' + str(k), k)
            negbin_k.value = np.log([self.nfailures[k], self.beta_a[k], self.beta_b[k]])
            negbin_k.connect_prior(mu, sigsqr)

        mu.initialize()
        mu_draws = np.zeros((self.niter, 3))
        for i in range(self.niter):
            mu_draws[i] = mu.random_draw()

        mu_low = np.percentile(mu_draws, 1.0, axis=0)
        mu_high = np.percentile(mu_draws, 99.0, axis=0)

        for j in range(3):
            # self.assertLess(mu_low[j], self.negbin_mu[j])
            # self.assertGreater(mu_high[j], self.negbin_mu[j])
            plt.hist(mu_draws[:, j], bins=100)
            plt.vlines(self.negbin_mu[j], plt.ylim()[0], plt.ylim()[1])
            plt.xlabel('Negbin Mu ' + str(j))
            plt.show()

        # now test prior mean with a transform
        mu = PriorMu('prior-mu', np.zeros(self.nbins), 10.0 * np.ones(self.nbins), transform=alpha_transform)
        sigsqr = PriorVar('prior-var', np.ones(self.nbins), np.ones(self.nbins), transform=alpha_transform)
        sigsqr.value = self.alpha_inverse_var

        mu.child_var = sigsqr
        for k in range(self.ncomponents):
            alpha_k = LogBinProbsAlpha(self.bin_counts, 'log-alpha-' + str(k), k)
            alpha_k.value = np.log(self.alpha[:, k])
            alpha_k.connect_prior(mu, sigsqr)

        mu.initialize()
        mu_draws = np.zeros((self.niter, self.nbins))
        for i in range(self.niter):
            mu_draws[i] = mu.random_draw()

        mu_low = np.percentile(mu_draws, 1.0, axis=0)
        mu_high = np.percentile(mu_draws, 99.0, axis=0)

        for j in range(self.nbins):
            # self.assertLess(mu_low[j], self.alpha_inverse_mu[j])
            # self.assertGreater(mu_high[j], self.alpha_inverse_mu[j])
            plt.hist(mu_draws[:, j], bins=100)
            plt.vlines(self.alpha_inverse_mu[j], plt.ylim()[0], plt.ylim()[1])
            plt.xlabel('Alpha Inverse Mu ' + str(j))
            plt.show()

    def test_prior_var(self):
        mu = PriorMu('prior-mu', np.zeros(self.nbins), 10.0 * np.ones(self.nbins), transform=alpha_transform)
        sigsqr = PriorVar('prior-var', np.ones(self.nbins), np.ones(self.nbins) / 10.0, transform=alpha_transform)
        mu.value = self.alpha_inverse_mu
        sigsqr.child_mean = mu

        for k in range(self.ncomponents):
            alpha_k = LogBinProbsAlpha(self.bin_counts, 'log-alpha-' + str(k), k)
            alpha_k.value = np.log(self.alpha[:, k])
            alpha_k.connect_prior(mu, sigsqr)

        sigsqr.initialize()
        var_draws = np.zeros((self.niter, self.nbins))
        for i in range(self.niter):
            var_draws[i] = sigsqr.random_draw()

        var_low = np.percentile(var_draws, 1.0, axis=0)
        var_high = np.percentile(var_draws, 99.0, axis=0)

        for j in range(self.nbins):
            self.assertLess(var_low[j], self.alpha_inverse_mu[j])
            self.assertGreater(var_high[j], self.alpha_inverse_mu[j])
            plt.hist(np.sqrt(var_draws[:, j]), bins=100)
            plt.vlines(np.sqrt(self.alpha_inverse_var[j]), plt.ylim()[0], plt.ylim()[1])
            plt.xlabel('Alpha Inverse Standard Deviation ' + str(j))
            plt.show()

    def test_prior_covar(self):

        mu = PriorMu('prior-mu', np.zeros(3), 10.0 * np.identity(3))
        sigsqr = PriorCovar('prior-covar', 1, np.identity(3) / 10.0)
        mu.value = self.negbin_mu
        sigsqr.child_mean = mu

        for k in range(self.ncomponents):
            negbin_k = MixLogNegBinPars(self.bin_counts.sum(axis=1), 'log-negbin-' + str(k), k)
            negbin_k.value = np.log([self.nfailures[k], self.beta_a[k], self.beta_b[k]])
            negbin_k.connect_prior(mu, sigsqr)

        sigsqr.initialize()
        covar_draws = np.zeros((self.niter, 3, 3))
        for i in range(self.niter):
            covar_draws[i] = sigsqr.random_draw()

        cov_low = np.percentile(covar_draws, 0.5, axis=0)
        cov_high = np.percentile(covar_draws, 99.5, axis=0)

        for j in range(3):
            for i in range(j, 3):
                print (j, i)
                self.assertLess(cov_low[j, i], self.negbin_covar[j, i])
                self.assertGreater(cov_high[j, i], self.negbin_covar[j, i])
                plt.hist(covar_draws[:, j, i], bins=100)
                plt.vlines(self.negbin_covar[j, i], plt.ylim()[0], plt.ylim()[1])
                plt.xlabel('Negbin Covar ' + str(j) + ', ' + str(i))
                plt.show()

    def test_components(self):
        pass

    def test_NegBinCounts(self):

        negbin = LogNegBinCounts(self.bin_counts.sum(axis=1), 'log-neg-bin')
        ram = RobustAdaptiveMetro(negbin, initial_covar=0.01, stop_adapting_iter=self.nburn)
        sampler = Sampler()
        sampler.add_step(ram)
        sampler.run(self.niter, nburn=self.nburn, verbose=True)

        samples = np.array(sampler.samples['log-neg-bin'])[:, 0]
        samples = pd.Series(samples, name='log-neg-bin')

        samples = np.exp(samples)
        lower = np.percentile(samples, 2.5)
        upper = np.percentile(samples, 97.5)

        self.assertGreater(upper, self.nfailures)
        self.assertLess(lower, self.nfailures)

        samples.plot(style='.')
        plt.plot(plt.xlim(), [self.nfailures] * 2, 'k')
        plt.show()
        samples.plot(kind='kde')
        plt.plot([self.nfailures] * 2, plt.ylim(), 'k')
        plt.show()

    def test_Concentration(self):

        conc = LogConcentration(self.bin_counts, 'log-c')
        gamma = LogBinProbsGamma(self.bin_counts, 'log-q')
        gamma.value = np.log(self.gamma)
        conc.connect_log_gamma(gamma)
        ram = RobustAdaptiveMetro(conc, initial_covar=0.01, stop_adapting_iter=self.nburn)
        sampler = Sampler()
        sampler.add_step(ram)
        sampler.run(self.niter, nburn=self.nburn, verbose=True)

        samples = np.array(sampler.samples['log-c'])[:, 0]
        samples = pd.Series(samples, name='log-c')

        samples = np.exp(samples)
        lower = np.percentile(samples, 2.5)
        upper = np.percentile(samples, 97.5)

        self.assertGreater(upper, self.concentration)
        self.assertLess(lower, self.concentration)

        samples.plot(style='.')
        plt.plot(plt.xlim(), [self.concentration] * 2, 'k')
        plt.show()
        samples.plot(kind='kde')
        plt.plot([self.concentration] * 2, plt.ylim(), 'k')
        plt.show()

    def test_BinProbs(self):
        conc = LogConcentration(self.bin_counts, 'log-c')
        gamma = LogBinProbsGamma(self.bin_counts, 'log-q')
        conc.value = np.log(self.concentration)
        conc.connect_log_gamma(gamma)
        ram = RobustAdaptiveMetro(gamma, initial_covar=0.01 * np.identity(self.nbins), stop_adapting_iter=self.nburn)
        sampler = Sampler()
        sampler.add_step(ram)
        sampler.run(self.niter, nburn=self.nburn, verbose=True)

        samples = np.array(sampler.samples['log-q'])
        samples = np.exp(samples)  # convert from log(gamma) to bin_probs
        samples /= samples.sum(axis=1)[:, np.newaxis]
        columns = ['bin_probs_' + str(i+1) for i in range(self.nbins)]
        samples = pd.DataFrame(samples, columns=columns)

        samples.plot(style='.')
        xlim = plt.xlim()
        for i in range(self.nbins):
            plt.plot(xlim, [self.bin_probs_parent[i]] * 2, 'k')
        plt.ylim(0, 1)
        plt.show()

        samples.plot(kind='kde')
        plt.xlim(0, 1)
        plt.show()


        sns.violinplot(samples)
        plt.yscale('log')
        plt.plot(range(1, self.nbins + 1), list(self.bin_probs_parent), 'ko')
        plt.show()

        lower = np.percentile(samples, 0.5, axis=0)
        upper = np.percentile(samples, 99.5, axis=0)

        for i in range(self.nbins):
            self.assertGreater(upper[i], self.bin_probs_parent[i])
            self.assertLess(lower[i], self.bin_probs_parent[i])

    def test_sampler(self):
        sampler = run_sampler(self.bin_counts, 10000, burniter=2500)

        print 'Acceptance rates:'
        for step in sampler.steps:
            print step.parameter.label, step.acceptance_rate()

        r_samples = np.exp(sampler.samples['log-nfailures'])
        c_samples = np.exp(sampler.samples['log-conc'])
        q_samples = np.array(sampler.samples['log-gamma'])
        q_samples = np.exp(q_samples)  # convert from log(gamma) to bin_probs
        q_samples /= q_samples.sum(axis=1)[:, np.newaxis]

        bin_cols = ['bin_probs_' + str(i+1) for i in range(self.nbins)]
        columns = ['nfailures', 'concentration'] + bin_cols
        samples = pd.DataFrame(np.column_stack((r_samples, c_samples, q_samples)), columns=columns)

        samples.plot(style='.', logy=True)
        xlim = plt.xlim()
        plt.plot(xlim, [self.concentration] * 2, 'k')
        plt.plot(xlim, [self.nfailures] * 2, 'k')
        for i in range(self.nbins):
            plt.plot(xlim, [self.bin_probs_parent[i]] * 2, 'k')
        plt.ylim(0, 1)
        plt.show()

        samples['nfailures'].plot(kind='kde')
        plt.show()
        samples['concentration'].plot(kind='kde')
        plt.show()
        samples[bin_cols].plot(kind='kde')
        plt.xlim(0, 1)
        plt.show()

        sns.violinplot(samples)
        plt.yscale('log')
        plt.plot(1, self.nfailures, 'ko')
        plt.plot(2, self.concentration, 'ko')
        plt.plot(range(3, self.nbins + 3), list(self.bin_probs_parent), 'ko')
        plt.show()

        lower = samples.quantile(0.005, axis=0)
        upper = samples.quantile(0.995, axis=0)

        self.assertGreater(upper['nfailures'], self.nfailures)
        self.assertLess(lower['nfailures'], self.nfailures)
        self.assertGreater(upper['concentration'], self.concentration)
        self.assertLess(lower['concentration'], self.concentration)

        for i, c in enumerate(bin_cols):
            self.assertGreater(upper[c], self.bin_probs_parent[i])
            self.assertLess(lower[c], self.bin_probs_parent[i])

