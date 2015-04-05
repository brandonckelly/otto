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
  int<lower=0> max_total_counts;
  int<lower=0> max_counts;
  
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
      lgamma_counts[j, i] <- lgamma(counts[i, j] + 1);  // matrices are stored column major, while arrays are row major
    }
    lgamma_total_counts[i] <- lgamma(total_counts[i] + 1);
  }

  max_total_counts <- max(total_counts);
  
  for (j in 1:nbins) {
    bin_probs_prior[j] <- 1.0;
  }
}

parameters {
  real<lower=0.0> concentration;  // concentration parameter for dirichlet prior on counts ratios
  simplex[nbins] bin_probs;  // histogram frequencies
}

model {
  real lgamma_concentration;
  vector[nbins] lgamma_concentration_bins;
  vector[max_total_counts + 1] lgamma_lookup_table_total_counts;
  vector[nbins] lgamma_lookup_table_counts[max_counts + 1];
  
  concentration ~ lognormal(0.0, 2.0);
  bin_probs ~ dirichlet(bin_probs_prior);
  
  // precompute gamma functions for efficiency
  lgamma_concentration <- lgamma(concentration);
  for (j in 1:nbins) {
    lgamma_concentration_bins[j] <- lgamma(concentration * bin_probs[j]);
  }
  for (i in 0:max_total_counts) {
    if (i == 0) {
      lgamma_lookup_table_total_counts[i + 1] <- lgamma(i + concentration);
    } else {
      lgamma_lookup_table_total_counts[i + 1] <- log(i - 1 + concentration) + lgamma_lookup_table_total_counts[i];
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
  for (i in 1:ntrain) {
    real logpost_temp;
    int total_counts_idx;
    logpost_temp <- 0.0;
    // Gamma(n_i + 1)
    logpost_temp <- logpost_temp + lgamma_total_counts[i];
    // Gamma(concentration)
    logpost_temp <- logpost_temp + lgamma_concentration;
    total_counts_idx <- total_counts[i] + 1;  // stan does not do zero-indexing
    // Gamma(n_i + concentration)
    logpost_temp <- logpost_temp - lgamma_lookup_table_total_counts[total_counts_idx];
    for (j in 1:nbins) {
      int counts_idx;
      counts_idx <- counts[i, j] + 1;  // stan does not do zero-indexing
      // Gamma(x_ij + concentration * bin_prob[j])
      logpost_temp <- logpost_temp + lgamma_lookup_table_counts[counts_idx][j];
      // Gamma(x_ij + 1)
      logpost_temp <- logpost_temp - lgamma_counts[j, i];
      // Gamma(concentration * bin_probs[j])
      logpost_temp <- logpost_temp - lgamma_concentration_bins[j];
    }
    increment_log_prob(logpost_temp);
  }
}