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
from copy import copy

def data_processing(df):
    df[df == '?'] = np.nan # Replace missing values by NAN values
    df_new = df.dropna(axis=0) # Remove missing values

    df_new['earnings'].replace({' <=50K':0,' >50K':1},inplace=True)
    df_new = df_new.drop('education-num',axis=1) # Redundant attribute
    df_new = df_new.drop('fnlwgt',axis=1) # Negative correlation.

    return df_new

def numerical_correlation(df_new):
    numerical = ['age','capital-loss','capital-gain','hours-per']
    for i in numerical:
        print(i,':',stats.pointbiserialr(df_new['earnings'],df_new[i])[0]) #Correlation between a binary variable and continuous variables

def categorical_correlation(df_encoded):

    def cross_tab(obs1, df):
        observed = pd.crosstab(obs1, df['earnings']) #Contingency table
        val = stats.chi2_contingency(observed)
        return(val[1])

    category = df_encoded.columns[df_encoded.dtypes!=object].tolist()[5:] #Categorical variables are encoded
    alpha = 0.01
    count = 0
    features = []
    dependent = []
    for i in category:
        p_value = cross_tab(df_encoded[i], df_encoded)
        if p_value <= alpha:
            count += 1
            features.append(i)
            dependent.append(i)
            dependent.append(p_value)
    print(dependent)

def one_hot_encoding(df):
    categorical_cols = df.columns[df.dtypes==object].tolist()
    return pd.get_dummies(df,columns=categorical_cols)

def normalization(df_encoded):
    columns_to_scale = ['age', 'capital-gain', 'capital-loss', 'hours-per']
    mms = preprocessing.MinMaxScaler()
    min_max_scaled_columns = mms.fit_transform(df_encoded[columns_to_scale])

    df_encoded['age'] = min_max_scaled_columns[:,0]
    df_encoded['capital-gain'] = min_max_scaled_columns[:,1]
    df_encoded['capital-loss'] = min_max_scaled_columns[:,2]
    df_encoded['hours-per'] = min_max_scaled_columns[:,3]

def split_samples(df_encoded):
    x = df_encoded.drop('earnings', axis=1)
    y = df_encoded['earnings']

    smote = SMOTE()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test
    x_train, y_train = smote.fit_resample(x_train, y_train)

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def tree_classifier(samples):
    dt = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=8,max_depth=10)
    dt.fit(samples["x_train"],samples["y_train"])
    return dt

def random_forest_classifier(samples):
    rf = RandomForestClassifier(min_samples_split=30)
    rf.fit(samples["x_train"],samples["y_train"])
    return rf

def predict(model, samples, print_=False):
    predictions = model.predict(samples["x_test"])

    if (print_): 
        print(classification_report(samples["y_test"], predictions))
        print(confusion_matrix(samples["y_test"], predictions))
        print(accuracy_score(samples["y_test"], predictions))
    
    return predictions

def save_tree_to_image(dt):
    dot_data = StringIO()
    export_graphviz(dt, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,feature_names = features,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('arvore.png')

def rich_proportions_of(attribute, samples, predictions):
    df = copy(samples["x_test"])
    df["earnings"] = samples["y_test"]
    rich_with_attribute_before = df[(df[attribute] == 1) & (df["earnings"] == 1)]
    proportion_before = len(rich_with_attribute_before) / len(df[df["earnings"] == 1])
    print(str(round(100 * proportion_before, 2)) + r"% of people with " + attribute + " earns more than 50K dollars per year, before training." )

    df_after = copy(samples["x_test"])
    df_after["earnings"] = predictions
    rich_with_attribute_after = df_after[(df_after[attribute] == 1) & (df_after["earnings"] == 1)]
    proportion_after = len(rich_with_attribute_after) / len(df_after[df_after["earnings"] == 1])
    print(str(round(100 * proportion_after, 2)) + r"% of people with " + attribute + " earns more than 50K dollars per year, after training." )

def main():
    df = pd.read_csv(r"adults_dataset/adult.csv")
    df.columns = [ 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                   'hours-per', 'native-country', 'earnings' ]

    df = data_processing(df)

    df_encoded = one_hot_encoding(df)
    
    normalization(df_encoded)

    samples = split_samples(df_encoded)
    


    # Decision tree
    dt = tree_classifier(samples)

    # Random Forest
    # rf = random_forest_classifier(samples)

    predictions = predict(dt, samples)

    rich_proportions_of("sex_ Male", samples, predictions)

if (__name__ == '__main__'):
    main()

