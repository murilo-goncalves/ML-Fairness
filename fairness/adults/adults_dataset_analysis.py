from io import StringIO
from os import name
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from sklearn import preprocessing, tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from collections import Counter
import pydotplus
from six import StringIO
from copy import copy

def name_columns(df):
    df.columns =  [ 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                   'hours-per', 'native-country', 'earnings' ]
    return df

def data_preprocessing(df):
    df[df == ' ?'] = np.nan # Replace missing values by NAN values
    df = df.dropna(axis=0) # Remove missing values

    df['earnings'].replace({' <=50K':0,' >50K':1},inplace=True)
    df['sex'].replace({' Female':0,' Male':1},inplace=True)
    df = df.drop('education-num',axis=1) # Redundant attribute
    df = df.drop('fnlwgt',axis=1)  # Redundant attribute
    df['native-country'] = np.where((df['native-country'] != ' United-States') & (df['native-country'] != ' Mexico'), ' Other', df['native-country'])

    return df

def fig_proportion_of_rich(df):
    earnings_by_sex = df[["sex", "earnings"]].groupby("sex").mean()
    plt.figure()
    fig_earnings_by_sex = plt.bar(["Female", "Male"], earnings_by_sex.earnings)
    plt.xlabel("Sex")
    plt.ylabel("Proportion of high earners by sex")
    plt.title("Proportion of high earners")

    earnings_by_race = df[["race", "earnings"]].groupby("race").mean()
    plt.figure()
    fig_earnings_by_race = plt.bar(["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"],
                                 earnings_by_race.earnings)
    plt.xlabel("Race")
    plt.ylabel("Proportion of high earners by race")
    plt.title("Proportion of high earners")

    df['age_group'] = pd.cut(df['age'],bins=[0, 25, 35, 60, 100],labels=['young','young-adult','adult','elderly'])
    earnings_by_age = df[["age_group", "earnings"]].groupby("age_group").mean()
    plt.figure()
    fig_earnings_by_age = plt.bar(["Young", "Young-Adult", "Adult", "Elderly"],
                                 earnings_by_age.earnings)
    plt.xlabel("Age group")
    plt.ylabel("Proportion of high earners by age group")
    plt.title("Proportion of high earners")

    earnings_by_country = df[["native-country", "earnings"]].groupby("native-country").mean()
    plt.figure()
    fig_earnings_by_country = plt.bar(["Mexico", "Other", "United States"],
                                 earnings_by_country.earnings)
    plt.xlabel("Native country")
    plt.ylabel("Proportion of high earners by native country")
    plt.title("Proportion of high earners")

    earnings_by_workclass = df[["workclass", "earnings"]].groupby("workclass").mean()
    plt.figure()
    fig_earnings_by_workclass = plt.bar(["Federal-gov", "Local-gov", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"],
                                 earnings_by_workclass.earnings)
    plt.xlabel("Workclass")
    plt.ylabel("Proportion of high earners by workclass")
    plt.title("Proportion of high earners")

    earnings_by_relationship = df[["relationship", "earnings"]].groupby("relationship").mean()
    plt.figure()
    fig_earnings_by_relationship = plt.bar(["Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"],
                                 earnings_by_relationship.earnings)
    plt.xlabel("Relationship")
    plt.ylabel("Proportion of high earners by relationship")
    plt.title("Proportion of high earners")

    # plt.show()

def numerical_correlation(df):
    numerical = ['age','capital-loss','capital-gain','hours-per']
    for i in numerical:
        print(i,':',stats.pointbiserialr(df['earnings'],df[i])[0]) #Correlation between a binary variable and continuous variables

def one_hot_encoding(df):
    categorical_cols = ["workclass", "education", "occupation", "race",
                        "relationship", "marital-status", "native-country",]
    return pd.get_dummies(df,columns=categorical_cols)

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

def normalization(df_encoded):
    columns_to_scale = ['age', 'capital-gain', 'capital-loss', 'hours-per']
    scaler = preprocessing.MinMaxScaler()
    min_max_scaled_columns = scaler.fit_transform(df_encoded[columns_to_scale])

    df_encoded['age'] = min_max_scaled_columns[:,0]
    df_encoded['capital-gain'] = min_max_scaled_columns[:,1]
    df_encoded['capital-loss'] = min_max_scaled_columns[:,2]
    df_encoded['hours-per'] = min_max_scaled_columns[:,3]

    # df_encoded['age'] = scaler.inverse_transform(min_max_scaled_columns)[:, [0]]

def split_samples(df_train, df_test):
    x_train = df_train.drop('earnings', axis=1)
    y_train = df_train['earnings']

    x_test = df_test.drop('earnings', axis=1)
    y_test = df_test['earnings']
 
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

def tree_classifier(samples):
    dt = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split = 8,max_depth = 10)
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

def proportion_of_rich(attribute, samples, predictions):
    df = copy(samples["x_test"])
    df["earnings"] = samples["y_test"]
    rich_with_attribute_before = df[(df[attribute] == 1) & (df["earnings"] == 1)]
    proportion_before = len(rich_with_attribute_before) / len(df["earnings"])
    print(str(round(100 * proportion_before, 2)) + r"% of people with " + attribute + " earn more than 50K dollars per year, before training." )

    df_after = copy(samples["x_test"])
    df_after["earnings"] = predictions
    rich_with_attribute_after = df_after[(df_after[attribute] == 1) & (df_after["earnings"] == 1)]
    proportion_after = len(rich_with_attribute_after) / len(df_after["earnings"])
    print(str(round(100 * proportion_after, 2)) + r"% of people with " + attribute + " earn more than 50K dollars per year, after training." )

def demographic_parity( df_test_encoded, predictions):
    dpd = demographic_parity_difference(df_test_encoded.earnings, predictions, sensitive_features=df_test_encoded.sex)
    dpr = demographic_parity_ratio(df_test_encoded.earnings, predictions, sensitive_features=df_test_encoded.sex)

    print(f"Demographic parity difference: {dpd:.3f}")
    print(f"Demographic parity ratio: {dpr:.3f}")

def main():
    df_data = pd.read_csv(r"C:\Users\marin\Desktop\UNICAMP\IC\ML-Fairness\fairness\adults\adults_dataset\adult_train.csv")
    df_data = name_columns(df_data)
    df_test = pd.read_csv(r"C:\Users\marin\Desktop\UNICAMP\IC\ML-Fairness\fairness\adults\adults_dataset\adult_test.csv")
    df_test = name_columns(df_test)

    df_data = data_preprocessing(df_data)
    df_test = data_preprocessing(df_test)

    # fig_proportion_of_rich(df_data)
    earnings_by_sex = df_data[["sex", "earnings"]].groupby("sex").mean()

    df_data_encoded = one_hot_encoding(df_data)
    df_test_encoded = one_hot_encoding(df_test)

    normalization(df_data_encoded)
    normalization(df_test_encoded)

    samples = split_samples(df_data_encoded, df_test_encoded)

    # # Decision tree
    # dt = tree_classifier(samples)
    
    # # Random Forest
    rf = random_forest_classifier(samples)

    predictions = predict(rf, samples, False)

    # proportion_of_rich("sex", samples, predictions)


if (__name__ == '__main__'):
    main()

