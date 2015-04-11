data {
  int<lower=0> ntrain;  // number of training samples
  int<lower=0> nbins;  // number of histogram bins, i.e. the number of features
  int<lower=0> counts[ntrain, nbins];  // histograms of features for each training data point
}

transformed data {
  int<lower=0> total_counts[ntrain];
  vector<lower=0>[nbins] bin_probs_prior;
  vector[ntrain] lgamma_total_counts;
  matrix[nbins, ntrain] lgamma_counts;
  real lgamma_total_counts_a;
  int<lower=0> max_total_counts;
  int<lower=0> max_counts;
  real<lower=0.0> negbin_a;  // beta prior on negative-binomial success probability
  real<lower=0.0> negbin_b;
  int<lower=0> all_counts;
  
  negbin_a <- 1.0;
  negbin_b <- 1.0;
  
  max_counts <- 0;
  for (i in 1:ntrain) {
    if (max(counts[i]) > max_counts) {
      max_counts <- max(counts[i]);
    }
  }
  
  for (i in 1:ntrain) {
    total_counts[i] <- 0;
    for (j in 1:nbins) {
      total_counts[i] <- total_counts[i] + counts[i, j];
      // matrices are stored column major, while arrays are row major
      lgamma_counts[j, i] <- lgamma(counts[i, j] + 1);  
    }
    lgamma_total_counts[i] <- lgamma(total_counts[i] + 1);
  }

  max_total_counts <- max(total_counts);
  
  for (j in 1:nbins) {
    bin_probs_prior[j] <- 1.0;
  }
  
  all_counts <- 0;
  for (i in 1:ntrain) {
    all_counts <- all_counts + total_counts[i];
  }
  lgamma_total_counts_a <- lgamma(all_counts + negbin_a);
}

parameters {
  real<lower=0.0> concentration;  // concentration parameter for dirichlet prior on counts ratios
  simplex[nbins] bin_probs;  // histogram frequencies
  real<lower=1.0> negbin_nfailures;  // parameter for negative-binomial distribution of total counts
}

model {
  real lgamma_concentration;
  vector[nbins] lgamma_concentration_bins;
  vector[max_total_counts + 1] lgamma_lookup_table_total_counts;
  vector[nbins] lgamma_lookup_table_counts[max_counts + 1];
  vector[max_total_counts + 1] lgamma_lookup_table_total_counts_negbin;
  real lgamma_negbin_nfailures;

  concentration ~ lognormal(0.0, 10.0);
  bin_probs ~ dirichlet(bin_probs_prior);
  // negbin_nfailures ~ lognormal(log(100.0), 10.0);
  negbin_nfailures ~ lognormal(log(100.0), 2.0);
  
  // precompute gamma functions for efficiency
  lgamma_concentration <- lgamma(concentration);
  lgamma_negbin_nfailures <- lgamma(negbin_nfailures);
  
  for (j in 1:nbins) {
    lgamma_concentration_bins[j] <- lgamma(concentration * bin_probs[j]);
  }
  
  for (i in 0:max_total_counts) {
    if (i == 0) {
      lgamma_lookup_table_total_counts[i + 1] <- lgamma(i + concentration);
      lgamma_lookup_table_total_counts_negbin[i + 1] <- lgamma(i + negbin_nfailures);
    } else {
      lgamma_lookup_table_total_counts[i + 1] <- log(i - 1 + concentration) + lgamma_lookup_table_total_counts[i];
      lgamma_lookup_table_total_counts_negbin[i + 1] <- log(i - 1 + negbin_nfailures) + 
        lgamma_lookup_table_total_counts_negbin[i];
    }
  }
  
  for (i in 0:max_counts) {
    if (i == 0) {
      for (j in 1:nbins) {
        lgamma_lookup_table_counts[i + 1][j] <- lgamma(i + concentration * bin_probs[j]);
      }
    } else {
      for (j in 1:nbins) {
        lgamma_lookup_table_counts[i + 1][j] <- log(i - 1 + concentration * bin_probs[j]) + 
          lgamma_lookup_table_counts[i][j];
      }
    }
  }
  
  // log-likelihood
  
  // contributions from negative-binomial component
  
  // Gamma(a + sum(total_counts))
  increment_log_prob(lgamma_total_counts_a);
  // Gamma(b + ndata * nfailures)
  increment_log_prob(lgamma(negbin_b + ntrain * negbin_nfailures));
  // Gamma(a + b + sum(total_counts) + ndata * nfailures)
  increment_log_prob(-lgamma(negbin_a + negbin_b + ntrain * negbin_nfailures + all_counts));

  for (i in 1:ntrain) {
    int total_counts_idx;
    // contribution from dirichlet-multinomial component
    
    // Gamma(n_i + 1)
    increment_log_prob(lgamma_total_counts[i]);
    // Gamma(concentration)
    increment_log_prob(lgamma_concentration);
    total_counts_idx <- total_counts[i] + 1;  // stan does not do zero-indexing
    // Gamma(n_i + concentration)
    increment_log_prob(-lgamma_lookup_table_total_counts[total_counts_idx]);
    for (j in 1:nbins) {
      int counts_idx;
      counts_idx <- counts[i, j] + 1;  // stan does not do zero-indexing
      // Gamma(x_ij + concentration * bin_prob[j])
      increment_log_prob(lgamma_lookup_table_counts[counts_idx][j]);
      // Gamma(x_ij + 1)
      increment_log_prob(-lgamma_counts[j, i]);
      // Gamma(concentration * bin_probs[j])
      increment_log_prob(-lgamma_concentration_bins[j]);
    }
    // more contributions from negative-binomial component
    
    // Gamma(n_i + nfailures)
    increment_log_prob(lgamma_lookup_table_total_counts_negbin[total_counts_idx]);
    // Gamma(n_i + 1)
    increment_log_prob(-lgamma_total_counts[i]);
    // Gamma(nfailures)
    increment_log_prob(-lgamma_negbin_nfailures);
  }
}