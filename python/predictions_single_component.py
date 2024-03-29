__author__ = 'brandonkelly'

from prediction import classify, rhat_class_predictions, get_prob_samples_all_classes, training_loss, \
    training_misclassification_rate
import numpy as np
import pandas as pd
import os


def combine_mcmc_output_single_component():
    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')

    files = [os.path.join(data_dir, 'single_component_samples_Class_' + str(k+1) + '.h5') for k in range(9)]

    df_list = [pd.read_hdf(f, 'df') for f in files]
    samples = pd.concat(df_list, axis=1, keys=['Class_' + str(k+1) for k in range(9)])
    return samples


if __name__ == "__main__":
    project_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto')
    data_dir = os.path.join(project_dir, 'data')

    # get the samples
    if os.path.isfile(os.path.join(data_dir, 'single_component_samples.h5')):
        samples = pd.read_hdf(os.path.join(data_dir, 'single_component_samples.h5'), 'df')
    else:
        samples = combine_mcmc_output_single_component()
        samples.to_hdf(os.path.join(data_dir, 'single_component_samples.h5'), 'df')

    # first get training error
    train_counts = pd.read_hdf(os.path.join(data_dir, 'train_processed.h5'), 'df')
    class_counts = train_counts.groupby('target').count()[train_counts.columns[0]]

    nsamples = 500
    nthin = len(samples) // nsamples
    samples = samples.iloc[::nthin]

    do_train = False
    do_test = True

    if do_train:
        print 'Getting target posterior probabilities...'
        # class_probs, rhat = classify(samples, train_counts, class_counts, compute_rhat=True, n_jobs=-1)
        # class_probs.to_hdf(os.path.join(data_dir, 'training_class_probs_single_component.h5'), 'df')
        # rhat.to_hdf(os.path.join(data_dir, 'training_rhat_single_component.h5'), 'df')

        class_probs = pd.read_hdf(os.path.join(data_dir, 'training_class_probs_single_component.h5'), 'df')

        print 'Computing the training loss...'
        loss = training_loss(class_probs, train_counts)
        print 'Training loss is', loss
        mrate = training_misclassification_rate(class_probs, train_counts)
        print 'Training misclassification rate is', mrate

    if do_test:
        print 'Computing test set predictions...'
        test_counts = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        test_class_probs, test_rhat = classify(samples, test_counts, class_counts, compute_rhat=True, n_jobs=-1)
        test_class_probs.to_hdf(os.path.join(data_dir, 'test_class_probs_single_component.h5'), 'df')
        test_rhat.to_hdf(os.path.join(data_dir, 'test_rhat_single_component.h5'), 'df')