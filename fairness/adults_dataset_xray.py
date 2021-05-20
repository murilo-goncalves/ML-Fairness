import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

    

def plot_count(df, attribute, hue=None):
    if (hue != None):
        sns.countplot(x=attribute, data=df, hue=hue)
    else:
        sns.countplot(x=attribute, data=df)
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


def main():
    sns.set(style='whitegrid', color_codes=True, rc={'figure.figsize':(11.7,8.27)})

    df = pd.read_csv(r"C:\Users\marin\Desktop\UNICAMP\IC\ML-Fairness\fairness\adults_dataset\adult.data")

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

    # discretizing age
    # df['age'] = ['young' if line < 25 else 
    #                     'young-adult' if line < 35 else
    #                     'adult' if line < 60 else
    #                     'elderly' for line in df['age']]

    categorical = [ 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'earnings', 'capital-gain', 'education' ]
    numerical = [ 'capital-loss', 'hours-per', 'education-num', 'fnlwgt', 'age' ]

    # relative_mean(df, 'sex', 'earnings', True)
    attributes = ['earnings', 'age', 'workclass', 'education-num', 'race', 'sex', 'native-country', 'hours-per']
    plot_correlation(df, categorical, attributes)

    plot_count(df, "earnings")

if (__name__ == '__main__'):
    main()

# count plot sex: more males than females on the dataset
# count plot earnings: much more people earning less than 50k
# box plot age, earnings: older people tend to earn more
# box plot sex, earnings: male tend to earn more
# box plot race, earnings: white and asian-pac-islander tend to earn more