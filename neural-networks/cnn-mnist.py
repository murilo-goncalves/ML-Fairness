import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import SGD

mndata = MNIST('data')

np.set_printoptions(precision=2, suppress=True)

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()
X_train, y_train = torch.from_numpy(np.array(X_train)/255), torch.from_numpy(np.array(y_train))
X_test, y_test = torch.from_numpy(np.array(X_test)/255), torch.from_numpy(np.array(y_test))

class CNN(Module):   
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = Sequential(
            Softmax(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = CNN()
optmizer = SGD(model.parameters(), lr=0.07)
