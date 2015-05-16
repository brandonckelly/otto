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
        np.random.seed(1234)

        self.ndata = 1000
        self.nbins = 10
        self.ncomponents = 10
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
        self.negbin_covar *= 9.0

        negbin_pars = np.random.multivariate_normal(self.negbin_mu, self.negbin_covar, self.ncomponents)
        self.nfailures = np.exp(negbin_pars[:, 0])
        self.beta_a = np.exp(negbin_pars[:, 1])
        self.beta_b = np.exp(negbin_pars[:, 2])

        self.alpha_inverse_mu = np.array([np.log(5)] + [0] * (self.nbins - 1))
        self.alpha_inverse_var = np.ones_like(self.alpha_inverse_mu) * 0.1 ** 2
        self.alpha_inverse_var * 36.0

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

                # mean_counts = self.bin_counts[k_idx].sum(axis=1).mean()
                # mean_true = self.nfailures[k] * self.beta_b[k] / (self.beta_a[k] - 1)
                # mean_bc = self.bin_counts[k_idx].mean(axis=0)
                # mean_true_alpha = mean_counts * self.alpha[:, k] / self.alpha[:, k].sum()
                # print k, mean_counts, mean_true
                # print mean_bc
                # print mean_true_alpha
                # print ''
                #

        self.niter = 7500
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

        labels = MixtureComponents('z', self.bin_counts)
        sticks = StickWeight('sticks')
        sticks.value = self.stick_weights
        labels.stick_weights = sticks

        for k in range(self.ncomponents):
            negbin_k = MixLogNegBinPars(self.bin_counts.sum(axis=1), 'log-negbin-' + str(k), k)
            negbin_k.value = np.log([self.nfailures[k], self.beta_a[k], self.beta_b[k]])
            # negbin_k.value = np.log([self.nfailures[j], self.beta_a[j], self.beta_b[j]])

            alpha_k = LogBinProbsAlpha(self.bin_counts, 'log-alpha-' + str(k), k)
            alpha_k.value = np.log(self.alpha[:, k])

            labels.add_component(negbin_k, alpha_k, k)

        labels.initialize()
        label_draws = np.empty((self.niter, self.ndata), dtype=int)
        for i in range(self.niter):
            if i % 1000 == 0:
                print i
            label_draws[i] = labels.random_draw()

        for k in range(self.ncomponents):
            nk = (label_draws == k).sum(axis=1)
            plt.hist(nk, bins=min(100, len(np.unique(nk)) * 2))
            nk_true = np.sum(self.component_labels == k)
            plt.vlines(nk_true, plt.ylim()[0], plt.ylim()[1], lw=3, color='red')
            plt.title('Component ' + str(k))
            plt.show()

        zcount = 0
        for i in range(self.ndata):
            zpick = np.bincount(label_draws[:, i]).argmax()
            zcount += zpick == self.component_labels[i]

        print 'Misclassification rate is', zcount / float(self.ndata)
        print 'No-information rate is', 1.0 - self.cluster_weights.max()

    def test_negbin(self):

        labels = MixtureComponents('z', self.bin_counts)
        labels.value = self.component_labels
        prior_mu = PriorMu('prior-mean', np.zeros(3), np.identity(3))
        prior_var = PriorCovar('prior-covar', 1, np.identity(3))
        prior_mu.value = self.negbin_mu
        prior_var.value = self.negbin_covar

        ram_steps = []

        for k in range(self.ncomponents):
            negbin_k = MixLogNegBinPars(self.bin_counts.sum(axis=1), 'log-negbin-' + str(k), k)

            alpha_k = LogBinProbsAlpha(self.bin_counts, 'log-alpha-' + str(k), k)
            alpha_k.value = np.log(self.alpha[:, k])

            labels.add_component(negbin_k, alpha_k, k)

            negbin_k.connect_prior(prior_mu, prior_var)

            negbin_k.initialize()
            ram_step = RobustAdaptiveMetro(negbin_k, initial_covar=np.identity(3) / 100.0, target_rate=0.3,
                                           stop_adapting_iter=self.nburn)
            ram_steps.append(ram_step)

        negbin_draws= np.empty((self.niter - self.nburn, 3, self.ncomponents))
        for i in range(self.niter):
            if i % 1000 == 0:
                print i
            for k, ram in enumerate(ram_steps):
                ram.do_step()
                if i >= self.nburn:
                    negbin_draws[i - self.nburn, :, k] = np.exp(ram.parameter.value)

        for k, ram in enumerate(ram_steps):
            print ""
            print k
            ram.report()

        r_low = np.percentile(negbin_draws[:, 0, :], 0.5, axis=0)
        r_high = np.percentile(negbin_draws[:, 0, :], 99.5, axis=0)
        a_low = np.percentile(negbin_draws[:, 1, :], 0.5, axis=0)
        a_high = np.percentile(negbin_draws[:, 1, :], 99.5, axis=0)
        b_low = np.percentile(negbin_draws[:, 2, :], 0.5, axis=0)
        b_high = np.percentile(negbin_draws[:, 2, :], 99.5, axis=0)

        for k in range(self.ncomponents):

            # self.assertLess(r_low[k], self.nfailures)
            # self.assertGreater(r_high[k], self.nfailures)
            # self.assertLess(a_low[k], self.beta_a)
            # self.assertGreater(a_high[k], self.beta_a)
            # self.assertLess(b_low[k], self.beta_b)
            # self.assertGreater(b_high[k], self.beta_b)
            fig = plt.figure()
            ax = plt.subplot(321)
            ax.hist(negbin_draws[:, 0, k], bins=25)
            ax.vlines(self.nfailures[k], plt.ylim()[0], plt.ylim()[1], lw=3, color='red')
            ax.set_xlabel('n failures')
            ax.set_title('Component ' + str(k))

            ax = plt.subplot(322)
            ax.hist(negbin_draws[:, 1, k], bins=25)
            ax.vlines(self.beta_a[k], plt.ylim()[0], plt.ylim()[1], lw=3, color='red')
            ax.set_xlabel('beta a')
            ax.set_title('Component ' + str(k))

            ax = plt.subplot(323)
            ax.hist(negbin_draws[:, 2, k], bins=25)
            ax.vlines(self.beta_b[k], plt.ylim()[0], plt.ylim()[1], lw=3, color='red')
            ax.set_xlabel('beta b')
            ax.set_title('Component ' + str(k))

            ax = plt.subplot(324)
            ax.plot(np.log(negbin_draws[:, 0, k]), np.log(negbin_draws[:, 1, k]), '.')
            ax.set_xlabel('log n failure')
            ax.set_ylabel('log beta a')
            ax.plot([np.log(self.nfailures[k])], np.log(self.beta_a[k]), 'ro')

            ax = plt.subplot(325)
            ax.plot(np.log(negbin_draws[:, 0, k]), np.log(negbin_draws[:, 2, k]), '.')
            ax.set_xlabel('log n failure')
            ax.set_ylabel('log beta b')
            ax.plot([np.log(self.nfailures[k])], np.log(self.beta_b[k]), 'ro')

            ax = plt.subplot(326)
            ax.plot(np.log(negbin_draws[:, 1, k]), np.log(negbin_draws[:, 2, k]), '.')
            ax.set_ylabel('log beta b')
            ax.set_xlabel('log beta a')
            ax.plot([np.log(self.beta_a[k])], np.log(self.beta_b[k]), 'ro')

            plt.show()
            plt.close()

    def test_alpha(self):

        labels = MixtureComponents('z', self.bin_counts)
        labels.value = self.component_labels
        prior_mu = PriorMu('prior-mean', np.zeros(self.nbins), np.identity(self.nbins))
        prior_var = PriorVar('prior-var', np.ones(self.nbins), np.ones(self.nbins))
        prior_mu.value = self.alpha_inverse_mu
        prior_var.value = self.alpha_inverse_var

        ram_steps = []

        for k in range(self.ncomponents):
            alpha_k = LogBinProbsAlpha(self.bin_counts, 'a' + str(k), k)

            negbin_k = MixLogNegBinPars(self.bin_counts, 'log-bnb-' + str(k), k)
            negbin_k.value = np.log([self.nfailures, self.beta_a, self.beta_b])

            labels.add_component(negbin_k, alpha_k, k)

            alpha_k.connect_prior(prior_mu, prior_var)

            alpha_k.initialize()
            ram_step = RobustAdaptiveMetro(alpha_k, initial_covar=np.identity(self.nbins) / 100.0, target_rate=0.15,
                                           stop_adapting_iter=self.nburn)
            ram_steps.append(ram_step)

        alpha_draws = np.empty((self.niter - self.nburn, self.nbins, self.ncomponents))
        for i in range(self.niter):
            if i % 1000 == 0:
                print i
            for k, ram in enumerate(ram_steps):
                ram.do_step()
                if i >= self.nburn:
                    alpha_draws[i - self.nburn, :, k] = ram.parameter.value

        for k, ram in enumerate(ram_steps):
            print ""
            print k
            ram.report()

        for k in range(self.ncomponents):
            df = pd.DataFrame(data=alpha_draws[:, :, k], columns=['bin_' + str(j) for j in range(self.nbins)])
            sns.boxplot(df)
            plt.plot(range(1, self.nbins+1), np.log(self.alpha[:, k]))
            plt.ylabel('log alpha')
            plt.title('Component ' + str(k))
            plt.show()
            plt.close()
