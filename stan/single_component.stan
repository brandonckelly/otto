data {
  int<lower=0> ntrain;  // number of training samples
  int<lower=0> nbins;  // number of histogram bins, i.e. the number of features
  matrix<lower=0>[nbins, ntrain] counts;  // histograms of features for each training data point
}

transformed data {
  vector<lower=0>[ntrain] total_counts;
  vector<lower=0>[nbins] bin_probs_prior;
  vector[ntrain] lgamma_total_counts;
  matrix[nbins, ntrain] lgamma_counts;
  
  for (i in 1:ntrain) {
    total_counts[i] <- 0;
    for (j in 1:nbins) {
      total_counts[i] <- total_counts[i] + counts[j, i];
      lgamma_counts[j, i] <- lgamma(counts[j, i] + 1);
    }
    lgamma_total_counts[i] <- lgamma(total_counts[i] + 1);
  }
  
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
  
  concentration ~ lognormal(0.0, 2.0);
  bin_probs ~ dirichlet(bin_probs_prior);
  
  // log-likelihood
  lgamma_concentration <- lgamma(concentration);  // pre-compute value for efficiency
  for (j in 1:nbins) {
    lgamma_concentration_bins[j] <- lgamma(concentration * bin_probs[j]);
  }
  
  for (i in 1:ntrain) {
    real logpost_temp;
    logpost_temp <- 0.0;
    logpost_temp <- logpost_temp + lgamma_total_counts[i];
    logpost_temp <- logpost_temp + lgamma(concentration);
    logpost_temp <- logpost_temp - lgamma(total_counts[i] + concentration);
    for (j in 1:nbins) {
      logpost_temp <- logpost_temp + lgamma(counts[j, i] + concentration * bin_probs[j]);
      logpost_temp <- logpost_temp - lgamma_counts[j, i];
      logpost_temp <- logpost_temp - lgamma_concentration_bins[j];
    }
    increment_log_prob(logpost_temp);
  }
}