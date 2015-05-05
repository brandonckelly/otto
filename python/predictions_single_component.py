__author__ = 'brandonkelly'

from prediction import classify, rhat_class_predictions, get_prob_samples_all_classes, training_loss
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

    print 'Getting target posterior probabilities...'
    class_probs = classify(samples, train_counts, compute_rhat=True)
    class_probs.to_hdf(os.path.join(data_dir, 'training_class_probs_single_component.h5'), 'df')

    print 'Computing the training loss...'
    loss = training_loss(class_probs, train_counts)
    print 'Training loss is', loss