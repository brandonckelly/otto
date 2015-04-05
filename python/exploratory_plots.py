__author__ = 'brandonkelly'

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def violin_plot_counts(df):
    sns.violinplot(df.counts, groupby=df.target)
    plt.show()


def box_plot_counts(df):
    sns.boxplot(df.counts, groupby=df.target)
    plt.show()



if __name__ == "__main__":

    data_dir = os.path.join(os.environ['HOME'], 'Projects', 'Kaggle', 'otto', 'data')
    df = pd.read_hdf(os.path.join(data_dir, 'train_processed.h5', 'df'))

    violin_plot_counts(df)
    box_plot_counts(df)