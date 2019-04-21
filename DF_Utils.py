import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# plot barplot with respect to increasing order of y
# parameter x must be of numerical data type
def barplot_increase(x, y, figX=10, figY=5):
    if not np.issubdtype(y.dtype, np.number):
        print("dependent variable is not of numerical data type!")
        return
    new_df = pd.concat([x, y], axis=1)
    attribute_x = new_df.columns[0]  # extract the column name of independent variable
    attribute_y = new_df.columns[1]  # extract the column name of dependent variable
    plt.figure(figsize=(figX, figY))
    result = new_df.groupby([attribute_x])[attribute_y].aggregate(np.mean).reset_index().sort_values(attribute_y)
    ax = sns.barplot(x=attribute_x, y=attribute_y, data=new_df, order=result[attribute_x])
    for item in ax.get_xticklabels():
        item.set_rotation(60)


# plot heatmap based on the dataframe. return quantitative and qualitative attributes respectively
def heatmap(df):
    quantitative = [f for f in df.columns if df.dtypes[f] != 'object']  # Numerical variable
    qualitative = [f for f in df.columns if df.dtypes[f] == 'object']  # Non-numerical variables variable
    sns.heatmap(df[quantitative].corr(), annot=True, cmap='cubehelix_r')
    return quantitative, qualitative


# normalize the whole dataframe (value type), and return the new dataframe, as well as the mean and standard deviation
# of each column in the old dataframe
def normalization(df):
    mean = df.mean()
    std = df.std()
    df = (df - mean) / std
    return df, mean, std


# normalize the whole dataframe (value type) using a certain mu and sigma
def normalization2(df, mu, sigma):
    df = (df - mu) / sigma
    return df
