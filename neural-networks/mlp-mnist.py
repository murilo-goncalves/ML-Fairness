from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from mnist import MNIST
import gzip
import numpy as np
import matplotlib.pyplot as plt

mndata = MNIST('data')

np.set_printoptions(precision=2, suppress=True)

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()
X_train, y_train = np.array(X_train)/255, np.array(y_train)
X_test, y_test = np.array(X_test)/255, np.array(y_test)


clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)

pred = clf.fit(X_train, y_train).predict(X_test)

print("Logistic Regression accuracy: ", accuracy_score(y_test, pred, normalize=True))


labels = [x for x in range(10)]
disp = plot_confusion_matrix(clf, X_test, y_test, display_labels=labels, cmap=plt.cm.Blues, normalize="true")
disp.ax_.set_title("chorro")

print("Confusion matrix:\n%s" % disp.confusion_matrix)
plt.show()