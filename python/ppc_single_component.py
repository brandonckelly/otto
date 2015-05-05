__author__ = 'brandonkelly'

from posterior_predictive_check import posterior_predictive_check
import pandas as pd
import os


if __name__ == "__main__":
    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')

    nfeatures = 93
    feature_columns = ['feat_' + str(i) for i in range(1, 94)]

    train_df = pd.read_hdf(os.path.join(data_dir, 'train_processed.h5'), 'df')

    class_labels = train_df['target'].unique()

    for target in class_labels:
        samples = pd.read_hdf(os.path.join(data_dir, 'single_component_samples_' + target + '.h5'), 'df')
        bin_counts = train_df.query("target == @target")[feature_columns]
        posterior_predictive_check(samples, bin_counts, target, nsamples=100)