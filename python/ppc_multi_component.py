__author__ = 'brandonkelly'

from posterior_predictive_check import posterior_predictive_check
import pandas as pd
import os
import multiprocessing


def ppc_helper(target):
    print target
    feature_columns = ['feat_' + str(i) for i in range(1, 94)]
    samples = pd.read_hdf(os.path.join(data_dir, 'multi_component_samples_' + target + '.h5'), 'df')
    bin_counts = train_df.query("target == @target")[feature_columns]
    posterior_predictive_check(samples, bin_counts, target, nsamples=100)


if __name__ == "__main__":
    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')

    nfeatures = 93
    feature_columns = ['feat_' + str(i) for i in range(1, 94)]

    train_df = pd.read_hdf(os.path.join(data_dir, 'train_processed.h5'), 'df')

    class_labels = train_df['target'].unique()

    class_labels = [class_labels[0]]

    n_jobs = 1
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()

    if n_jobs == 1:
        map(ppc_helper, class_labels)
    else:
        pool = multiprocessing.Pool(n_jobs)
        pool.map(ppc_helper, class_labels)
