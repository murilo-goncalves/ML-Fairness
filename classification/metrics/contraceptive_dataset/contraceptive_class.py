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

df_contraceptive = pd.read_csv('cmc.data')
df_contraceptive.columns = (["Wife's age",                   
                             "Wife's education",             
                             "Husband's education",          
                             "Number of children ever born", 
                             "Wife's religion", 
                             "Wife's now working?",          
                             "Husband's occupation",         
                             "Standard-of-living index",     
                             "Media exposure",               
                             "Contraceptive method used"])

# df_contraceptive.hist()
# sns.boxplot(x="Contraceptive method used", y="Wife's age", data=df_contraceptive)
# plt.show()

cols = [col for col in df_contraceptive.columns if col is not "Contraceptive method used"]
data = df_contraceptive[cols]
target = df_contraceptive['Contraceptive method used']
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.20, random_state = 10)

# data_train.info()

logReg = KNeighborsClassifier()
pred = logReg.fit(data_train, target_train).predict(data_test)

print("Logistic Regression accuracy: ", accuracy_score(target_test, pred, normalize=True))

visualizer = ClassificationReport(logReg)
visualizer.fit(data_train, target_train)
visualizer.score(data_test, target_test) 
visualizer.show()
