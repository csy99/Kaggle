import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# class Distribution:

#  Check which distribution this attribute follows so that we can do transformation before performing regression.
def check_dist(x):
    fig = plt.figure(figsize=(20, 40))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("Johnson SU")
    sns.distplot(x, kde=False, fit=st.johnsonsu)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Normal")
    sns.distplot(x, kde=False, fit=st.norm)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("Log Normal")
    sns.distplot(x, kde=False, fit=st.lognorm)


# apply johnson transformation
def johnson_transform(x):
    gamma, eta, epsilon, lbda = st.johnsonsu.fit(x)
    yt = gamma + eta * np.arcsinh((x - epsilon) / lbda)
    return yt, gamma, eta, epsilon, lbda


# apply inverse of johnson transformation
def johnson_inverse(y, gamma, eta, epsilon, lbda):
    return lbda * np.sinh((y - gamma) / eta) + epsilon


# # plot distribution of data in a dataframe
# def dist_graph(df):
#     # quantitative = [f for f in df if df[f] != 'object']
#     # qualitative = [f for f in df if df[f] == 'object']
#     # f
#     g = sns.FacetGrid(df, col_wrap=2, sharex=False, sharey=False)
#     g = g.map(sns.distplot)
