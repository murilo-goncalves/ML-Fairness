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

sns.set(style='whitegrid', color_codes=True, rc={'figure.figsize':(11.7,8.27)})

df_abalone = pd.read_csv('abalone.data')
df_abalone.columns = ['sex', 'length', 'diameter', 'height', 'whole weight', 'sucked weight', 'viscera weight', 'shell weight', 'rings']

sns.pairplot(data=df_abalone, hue='rings', palette='gist_heat', dropna=True)
# plt.show()

# print(df_abalone.describe())

le = preprocessing.LabelEncoder()
df_abalone['sex'] = le.fit_transform(df_abalone['sex'])

cols = [col for col in df_abalone.columns if col is not "rings"]
data = df_abalone[cols]
target = df_abalone['rings']
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.20, random_state = 10)

# data_train.info()

logReg = LogisticRegression()
pred = logReg.fit(data_train, target_train).predict(data_test)

print("Logistic Regression accuracy: ", accuracy_score(target_test, pred, normalize=True))

visualizer = PrecisionRecallCurve(logReg)
visualizer.fit(data_train, target_train)
visualizer.score(data_test, target_test) 
visualizer.show()