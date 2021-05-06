import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

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

    df[df == '?'] = np.nan #Replace missing values by NAN values
    df_new = df.dropna(axis=0) #Remove missing values

    df_new['earnings'].replace({'<=50K':0,'>50K':1},inplace=True)
    df_new = df_new.drop('education-num',axis=1) #Redundant attribute

    # numerical = ['age','capital-loss','capital-gain','hours-per','fnlwgt']
    # for i in numerical:
    #     print(i,':',stats.pointbiserialr(df_new['earnings'],df_new[i])[0]) #Correlation between a binary variable and continuous variables

    df_new = df_new.drop('fnlwgt',axis=1) #Negative correlation.

    def cross_tab(obs1=[]):
        observed=pd.crosstab(obs1,adult_new['earnings'])
        val=stats.chi2_contingency(observed)
        return(val[1])

    category = df_new.columns[df_new.dtypes!=object].tolist()[5:] #Categorical variables are encoded
    alpha = 0.01
    count = 0
    features = []
    for i in category:
       p_value = cross_tab(df_new[i])
       if p_value <= alpha:
           count += 1
           features.append(i)
    
    categorical_cols = df_new.columns[df_new.dtypes==object].tolist()
    df_new = pd.get_dummies(df_new,columns=categorical_cols)

    columns_to_scale = ['age', 'capital-gain', 'capital-loss', 'hours-per']
    mms = MinMaxScaler()
    min_max_scaled_columns = mms.fit_transform(df_new[columns_to_scale])

    df_new['age'],df_new['capital-gain'],df_new['capital-loss'],df_new['hours-per']= min_max_scaled_columns[:,0],min_max_scaled_columns[:,1], min_max_scaled_columns[:,2],min_max_scaled_columns[:,3]

if (__name__ == '__main__'):
    main()

