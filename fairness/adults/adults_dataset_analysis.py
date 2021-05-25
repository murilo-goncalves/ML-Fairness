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

def cross_tab(obs1, df):
        observed = pd.crosstab(obs1, df['earnings']) #Contingency table
        val = stats.chi2_contingency(observed)
        return(val[1])

def categorical_correlation(df_encoded):
    category = df_encoded.columns[df_encoded.dtypes!=object].tolist()[5:] #Categorical variables are encoded
    alpha = 0.01
    count = 0
    features = []
    for i in category:
        p_value = cross_tab(df_encoded[i], df_encoded)
        if p_value <= alpha:
           count += 1
           features.append(i)

def one_hot_encoding(df):
    categorical_cols = df.columns[df.dtypes==object].tolist()
    df_encoded = pd.get_dummies(df,columns=categorical_cols)
    

def main():
    df = pd.read_csv(r"adults_dataset/adult.csv")
    df.columns = [ 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                   'hours-per', 'native-country', 'earnings' ]

    df = data_processing(df)
    
    #NORMALIZATION
    columns_to_scale = ['age', 'capital-gain', 'capital-loss', 'hours-per']
    mms = preprocessing.MinMaxScaler()
    min_max_scaled_columns = mms.fit_transform(df_encoded[columns_to_scale])

    df_encoded['age'],df_encoded['capital-gain'],df_encoded['capital-loss'],df_encoded['hours-per'] = min_max_scaled_columns[:,0],min_max_scaled_columns[:,1], min_max_scaled_columns[:,2],min_max_scaled_columns[:,3]

    #PREDICTION
    x = df_encoded.drop('earnings', axis=1)
    y = df_encoded['earnings']

    smote = SMOTE()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test
    x_train, y_train = smote.fit_resample(x_train, y_train)

    #Decision tree
    # Create Decision Tree classifer object
    clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=8,max_depth=10)

    # Train Decision Tree Classifer
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    
    features = list(df_encoded.columns[1:])


    #Random Forest
    # rf = RandomForestClassifier(min_samples_split=30)

    # rf.fit(x_train,y_train)
    # predictions = rf.predict(x_test)

    # print(classification_report(y_test,predictions))
    # print(confusion_matrix(y_test,predictions))
    # print(accuracy_score(y_test,predictions))


    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data,
    #                 filled=True, rounded=True,
    #                 special_characters=True,feature_names = features,class_names=['0','1'])
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # graph.write_png('arvore.png')

if (__name__ == '__main__'):
    main()

