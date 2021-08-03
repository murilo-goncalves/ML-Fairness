import sys

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    true_positive_rate,
    false_positive_rate,
    true_negative_rate,
    false_negative_rate
)

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.datasets import StandardDataset

def name_columns(df):
    df.columns =  [ 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                   'hours-per', 'native-country', 'earnings' ]
    return df

def data_preprocessing(df):
    df = df.replace(' ?', np.nan, inplace=False) # Replace missing values by NAN values
    df = df.dropna(axis=0) # Remove missing values

    df['earnings'].replace({' <=50K':0,' >50K':1},inplace=True)
    df['sex'].replace({' Female':0,' Male':1},inplace=True)
    df = df.drop('education-num',axis=1) # Redundant attribute
    df = df.drop('fnlwgt',axis=1)  # Redundant attribute
    df['native-country'] = np.where((df['native-country'] != ' United-States') & (df['native-country'] != ' Mexico'), ' Other', df['native-country'])

    return df

def fig_proportion_of_rich(df, attribute, show=False):
    if (attribute == "age"):
        df['age_group'] = pd.cut(df['age'],bins=[0, 25, 35, 60, 100],labels=['young','young-adult','adult','elderly'])
        earnings= df[["age_group", "earnings"]].groupby("age_group").mean()
        plt.bar(["Young", "Young-Adult", "Adult", "Elderly"], earnings.earnings)
        df.drop('age_group', axis=1, inplace=True)

    else:
        earnings = df[[attribute, "earnings"]].groupby(attribute).mean()
        
        values = df[attribute].unique()
        values = list(map(str, values))
        values.sort()
        plt.bar(values, earnings.earnings)

    plt.xlabel(attribute)
    plt.ylabel(f"Proportion of high earners by {attribute}")
    plt.title("Proportion of high earners")

    if (show):
        plt.show()

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

def split_samples(df_train, df_test):
    x_train = df_train.drop('earnings', axis=1)
    y_train = df_train['earnings']

    x_test = df_test.drop('earnings', axis=1)
    y_test = df_test['earnings']
 
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

def random_forest_classifier(samples):
    model = RandomForestClassifier(min_samples_split=30)
    model.fit(samples["x_train"],samples["y_train"])
    return model

def predict(model, samples, print_=False):
    predictions = model.predict(samples["x_test"])

    if (print_): 
        print(classification_report(samples["y_test"], predictions))
        print(confusion_matrix(samples["y_test"], predictions))
        print(accuracy_score(samples["y_test"], predictions))
    
    return predictions

def proportion_of_rich(attribute, samples, predictions, print_=False):
    df = copy(samples["x_test"])
    df["earnings"] = samples["y_test"]
    rich_with_attribute_before = df[(df[attribute] == 1) & (df["earnings"] == 1)]
    proportion_before = len(rich_with_attribute_before) / len(df[(df[attribute] == 1)])
    
    df_after = copy(samples["x_test"])
    df_after["earnings"] = predictions
    rich_with_attribute_after = df_after[(df_after[attribute] == 1) & (df_after["earnings"] == 1)]
    proportion_after = len(rich_with_attribute_after) / len(df[(df[attribute] == 1)])

    if print_:
        print(str(round(100 * proportion_before, 2)) + r"% of people with " + attribute + " earn more than 50K dollars per year, before training." )
        print(str(round(100 * proportion_after, 2)) + r"% of people with " + attribute + " earn more than 50K dollars per year, after training." )

def gender_performance(df_test_encoded, predictions, print_=False):
    predictions_m = []
    predictions_f = []
    df_test_encoded_m = []
    df_test_encoded_f = []
    tamanho = len(df_test_encoded['sex'])

    for i in range(tamanho):
        if df_test_encoded['sex'].iloc[i] == 1:
            df_test_encoded_m.append(df_test_encoded['earnings'].iloc[i])
            predictions_m.append(predictions[i])
        else:
            df_test_encoded_f.append(df_test_encoded['earnings'].iloc[i])
            predictions_f.append(predictions[i])
  

    true_positive_m = true_positive_rate(df_test_encoded_m, predictions_m, pos_label=1)
    false_positive_m = false_positive_rate(df_test_encoded_m, predictions_m, pos_label=1)
    true_negative_m = true_negative_rate(df_test_encoded_m, predictions_m, pos_label=1)
    false_negative_m = false_negative_rate(df_test_encoded_m, predictions_m, pos_label=1)

    true_positive_f = true_positive_rate(df_test_encoded_f, predictions_f, pos_label=1)
    false_positive_f = false_positive_rate(df_test_encoded_f, predictions_f, pos_label=1)
    true_negative_f = true_negative_rate(df_test_encoded_f, predictions_f, pos_label=1)
    false_negative_f = false_negative_rate(df_test_encoded_f, predictions_f, pos_label=1)

    if print_:
        print("True Positive Rate for Male:", true_positive_m)
        print("True Positive Rate for Female:", true_positive_f)
        print("False Positive Rate for Male:", false_positive_m)
        print("False Positive Rate for Female:", false_positive_f)
        print("True Negative Rate for Male:", true_negative_m)
        print("True Negative Rate for Female:", true_negative_f)
        print("False Negative Rate for Male:", false_negative_m)
        print("False Negative Rate for Female:", false_negative_f)

def demographic_parity(df_test_encoded, predictions, print_=False):
    dpd_sex = demographic_parity_difference(df_test_encoded.earnings, predictions, sensitive_features=df_test_encoded.sex)
    dpr_sex = demographic_parity_ratio(df_test_encoded.earnings, predictions, sensitive_features=df_test_encoded.sex)

    if (print_):
        print(f"Demographic parity difference sex:", dpd_sex)
        print(f"Demographic parity ratio sex:", dpr_sex)

def equalized_odds(df_test_encoded, predictions, print_=False):
    eod_sex = equalized_odds_difference(df_test_encoded.earnings, predictions, sensitive_features=df_test_encoded.sex)
    eor_sex = equalized_odds_ratio(df_test_encoded.earnings, predictions, sensitive_features=df_test_encoded.sex)

    if (print_):
        print(f"equalised odds difference sex: {eod_sex:.3f}")
        print(f"equalised odds ratio sex: {eor_sex:.3f}")

def split_samples_fair(train_sds, test_sds, test_sds_pred):
    x_train_fair = train_sds.features
    y_train_fair = train_sds.labels.flatten()
    x_test_fair = test_sds_pred.features
    y_test_fair = test_sds.labels.flatten()

    return {"x_train": x_train_fair, "y_train": y_train_fair, "x_test": x_test_fair, "y_test": y_test_fair}

def logistic_regression(test_sds_transf):
    x = test_sds_transf.features
    y = test_sds_transf.labels.flatten()
    loreg = LogisticRegression(max_iter=10000) # initialize the model
    loreg.fit(x, y, sample_weight=test_sds_transf.instance_weights) # fit the model
    return loreg

def predict_fair(model, samples, print_=False):
    predictions = model.predict_proba(samples["x_test"])[:, 1]
    test_pred = predictions > 0.5
  
    return predictions, test_pred

def main(argv):
    df_data = pd.read_csv(r"adults_dataset/adult_train.csv")
    df_data = name_columns(df_data)
    df_test = pd.read_csv(r"adults_dataset/adult_test.csv")
    df_test = name_columns(df_test)

    df_data = data_preprocessing(df_data)
    df_test = data_preprocessing(df_test)

    # fig_proportion_of_rich(df_test, argv[1], False)

    df_data_encoded = one_hot_encoding(df_data)
    df_test_encoded = one_hot_encoding(df_test)

    normalization(df_data_encoded)
    normalization(df_test_encoded)

    samples = split_samples(df_data_encoded, df_test_encoded)
    
    model = random_forest_classifier(samples)

    predictions = predict(model, samples, False)

    # proportion_of_rich(argv[2], samples, predictions, False)

    gender_performance(df_test_encoded, predictions)
    demographic_parity(df_test_encoded, predictions)
    equalized_odds(df_test_encoded, predictions)

    #Kamiran and Calders
    train_sds = StandardDataset(df_data_encoded, label_name="earnings", favorable_classes=[1], 
                                protected_attribute_names=["sex"], privileged_classes=[[1]])

    test_sds = StandardDataset(df_test_encoded, label_name="earnings", favorable_classes=[1],
                               protected_attribute_names=["sex"], privileged_classes=[[1]])

    privileged_groups = [{"sex": 1.0}]
    unprivileged_groups = [{"sex": 0.0}]

    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    RW.fit(train_sds)

    test_sds_pred = test_sds.copy(deepcopy=True)
    test_sds_transf = RW.transform(test_sds)

    samples_fair = split_samples_fair(train_sds, test_sds, test_sds_pred)
    
    model_fair = logistic_regression(test_sds_transf)

    predictions_fair, test_pred = predict_fair(model_fair, samples_fair, True)
    test_pred = test_pred.astype(int)

    dpd = demographic_parity_difference(
        df_test_encoded.earnings, test_pred, sensitive_features=df_test_encoded.sex)

    print(f"Model demographic parity difference:", dpd)

if (__name__ == '__main__'):
    main(sys.argv)

