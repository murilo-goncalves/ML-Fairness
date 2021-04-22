import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, PrecisionRecallCurve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# TESTAR APENAS 8000 DADOS <= 50K, FAZER UM COMITÊ

sns.set(style='whitegrid', color_codes=True, rc={'figure.figsize':(11.7,8.27)})

df_adults = pd.read_csv("adult.data")
df_adults.columns = ['age',
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
                   'earnings']

categorical = ['education', 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'earnings']
numerical = ['education-num', 'capital-gain', 'capital-loss', 'hours-per']

sns.pairplot(data=df_adults, hue='earnings', palette='gist_heat', dropna=True)
sns.boxplot(data=df_adults, x='earnings', y='age')
sns.countplot(x='earnings', data=df_adults)
df_adults[numerical].hist()
plt.show()

le = preprocessing.LabelEncoder()
for i in range(9):
    df_adults[categorical[i]] = le.fit_transform(df_adults[categorical[i]])

# print(df_adults.head())

cols = [col for col in df_adults.columns if col != "earnings"]
data = df_adults[cols]
target = df_adults['earnings']
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.20, random_state = 10)


logReg = KNeighborsClassifier()
pred = logReg.fit(data_train, target_train).predict(data_test)

print("Logistic Regression accuracy: ", accuracy_score(target_test, pred, normalize=True))

visualizer = ClassificationReport(logReg)
visualizer.fit(data_train, target_train)
visualizer.score(data_test, target_test) 
visualizer.show()
