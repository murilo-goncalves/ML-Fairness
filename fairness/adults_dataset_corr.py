import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from sklearn import preprocessing, tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    df = pd.read_csv(r"C:\Users\marin\Desktop\UNICAMP\IC\ML-Fairness\fairness\adults_dataset\adult.csv")
    df.columns = [ 'age',
                'workclass',
                'fnlwgt',
                'education',
                'education-num',
                'marital-status',
                'occupation',
                'relationship',
                'race',
                'sex',
                'capital-gain',
                'capital-loss',
                'hours-per',
                'native-country',
                'earnings' ]

    #DATA PREPROCESING
    df[df == '?'] = np.nan #Replace missing values by NAN values
    df_new = df.dropna(axis=0) #Remove missing values

    df_new['earnings'].replace({' <=50K':0,' >50K':1},inplace=True)
    df_new = df_new.drop('education-num',axis=1) #Redundant attribute
    df_new = df_new.drop('fnlwgt',axis=1) #Negative correlation.

    #NUMERICAL CORRELATION
    numerical = ['age','capital-loss','capital-gain','hours-per']
    for i in numerical:
        print(i,':',stats.pointbiserialr(df_new['earnings'],df_new[i])[0]) #Correlation between a binary variable and continuous variables

    #CATEGORICAL CORRELATION
    def cross_tab(obs1=[]):
        observed = pd.crosstab(obs1,df_new['earnings']) #Contingency table
        val = stats.chi2_contingency(observed)
        return(val[1])

    categorical_cols = df_new.columns[df_new.dtypes==object].tolist()
    df_new1 = pd.get_dummies(df_new,columns=categorical_cols)

    category = df_new1.columns[df_new1.dtypes!=object].tolist()[5:] #Categorical variables are encoded
    alpha = 0.01
    count = 0
    features = []
    dependent = []
    for i in category:
        p_value = cross_tab(df_new1[i])
        if p_value <= alpha:
           count += 1
           features.append(i)
           dependent.append(i)
           dependent.append(p_value)

    # print(dependent)

    #NORMALIZATION
    columns_to_scale = ['age', 'capital-gain', 'capital-loss', 'hours-per']
    mms = MinMaxScaler()
    min_max_scaled_columns = mms.fit_transform(df_new1[columns_to_scale])

    df_new1['age'],df_new1['capital-gain'],df_new1['capital-loss'],df_new1['hours-per'] = min_max_scaled_columns[:,0],min_max_scaled_columns[:,1], min_max_scaled_columns[:,2],min_max_scaled_columns[:,3]

    #PREDICTION
    x = df_new1.drop('earnings', axis=1)
    y = df_new1['earnings']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test
    x_train, y_train = SMOTE().fit_resample(x_train, y_train)

    #Decision tree
    # Create Decision Tree classifer object
    clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=8,max_depth=10)

    # Train Decision Tree Classifer
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)

    features = list(df_new1.columns[1:])
    print(feautres)


    #Random Forest
    # rf=RandomForestClassifier(min_samples_split=30)

    # rf.fit(x_train,y_train)
    # predictions = rf.predict(x_test)

    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    print(accuracy_score(y_test,predictions))

if (__name__ == '__main__'):
    main()

