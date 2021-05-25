from io import StringIO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from sklearn import preprocessing, tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pydotplus
from six import StringIO

def plot_correlation(df, categorical, attributes):
    size = len(attributes)

    le = preprocessing.LabelEncoder()
    for category in categorical:
        df[category] = le.fit_transform(df[category])

    df = df[attributes]

    corr = df.corr()
    # _, ax = plt.subplots(figsize=(size,size))
    #ax.grid(False)
    #ax.matshow(corr)
    #plt.xticks(range(len(corr.columns)),corr.columns)
    #plt.yticks(range(len(corr.columns)),corr.columns)
    sns.heatmap(corr, annot=True)
    plt.show()

def plot_hist(df, attribute):
    df[attribute].hist()
    plt.show()

def plot_box(df, catAttribute, numAttribute):
    sns.boxplot(data=df, x=catAttribute, y=numAttribute)
    plt.show()

def relative_mean(df, catAttribute, numAttribute, isBothCat=False):
    if (isBothCat):
        le = preprocessing.LabelEncoder()
        df[numAttribute] = le.fit_transform(df[numAttribute])
    print(df[[catAttribute, numAttribute]].groupby([catAttribute]).mean())

def plot_count(df, attribute, hue=None):
    if (hue != None):
        sns.countplot(x=attribute, data=df, hue=hue)
    else:
        sns.countplot(x=attribute, data=df)
    plt.show()

def subplot_categs(dfs, titles, category, fignum=1):
    plt.figure(fignum, figsize=(12, 6))
    number_of_dfs = len(titles)
    first_axis = None
    for df_index, df in enumerate(dfs):
        title = titles[df_index]
        uniques = list(sorted(df[category].unique()))
        counts = [df[df[category]==value].shape[0] for value in uniques]
        size = len(uniques)
        xcoords = list(range(1, size+1))
        if df_index == 0:
            first_axis =plt.subplot(1, 2, df_index+1)
        else:
            new_axis = plt.subplot(1, 2, df_index + 1, sharey=first_axis)
        plt.bar(xcoords, counts)
        plt.xticks(xcoords, uniques, rotation='vertical' if size >= 5 else 'horizontal')
        plt.title((title if title else ''))
        plt.tight_layout()