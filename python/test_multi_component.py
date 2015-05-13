__author__ = 'brandonkelly'

__author__ = 'brandonkelly'

import unittest
import numpy as np
from bck_mcmc.sampler import Sampler
from bck_mcmc.steps import RobustAdaptiveMetro
from parameters import LogNegBinCounts, LogBinProbsGamma, LogConcentration
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from run_single_component_sampler import run_sampler
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
        self.ncomponents = 4
        self.bin_counts = np.zeros((self.ndata, self.nbins), dtype=np.int)

        self.dp_concentration = 1.0
        self.stick_weights = np.random.beta(1.0, self.dp_concentration)
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
        self.nfailures = np.exp(negbin_pars[0])
        self.beta_a = np.exp(negbin_pars[1])
        self.beta_b = np.exp(negbin_pars[2])

        self.alpha_mu = np.array([np.log(5)] + [0] * (self.nbins - 1))
        self.alpha_var = np.empty_like(self.alpha_mu).fill(0.1 ** 2)

        self.alpha = np.exp(self.alpha_mu +
                            np.sqrt(self.alpha_var) * np.random.standard_normal((self.nbins, self.ncomponents)))

        for k in range(self.ncomponents):
            k_idx = np.where(self.component_labels == k)[0]
            nk = len(k_idx)
            total_counts = bnb_sample(self.nfailures[k], self.beta_a[k], self.beta_b[k], nk)
            probs = np.random.dirichlet(self.alpha[:, k], nk)
            for i in range(nk):
                self.bin_counts[k_idx[i]] = np.random.multinomial(total_counts[i], probs[i])

        self.niter = 10000
        self.nburn = 2500

    def test_dp_concentration(self):
        pass

    def test_stick_weights(self):
        pass

    def test_prior_mu(self):
        pass

    def test_prior_var(self):
        pass

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

