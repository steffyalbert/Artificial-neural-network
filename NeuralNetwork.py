import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import random
np.seterr(divide='ignore', invalid='ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dSigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = s * (1 - s)
    return dZ

def initializeWeight(layer):
    W = np.random.rand(dim[layer + 1], dim[layer]) * np.sqrt(2 / dim[layer])
    return W

def forward(W1, W2, W3, X):
    # Z is WX
    Z1 = np.dot(W1, X.T)
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1)
    A2 = sigmoid(Z2)
    Z3 = np.dot(W3, A2)
    A3 = sigmoid(Z3)
    return Z1, Z2, Z3, A1, A2, A3

def backward(Z3, Z2, Z1, A1, A2, X, y_train, Yh):
    dLoss_Yh = - (np.divide(y_train, Yh)) + (np.divide(1 - y_train, 1 - Yh))
    sigmoid = dSigmoid(Z3)
    dLoss_Z3 = np.multiply(dLoss_Yh.T, sigmoid[0])
    dLoss_Z3 = dLoss_Z3.reshape(-1, 1)
    dLoss_A2 = np.dot(W3.T, dLoss_Z3.T)
    dLoss_W3 = np.dot(dLoss_Z3.T, A2.T)
    dLoss_Z2 = np.multiply(dLoss_A2.T, dSigmoid(Z2).T)
    dLoss_A1 = np.dot(W2, dLoss_Z2.T)
    dLoss_W2 = np.dot(dLoss_Z2.T, A1.T)
    dLoss_Z1 = np.multiply(dLoss_A1.T, dSigmoid(Z1).T)
    dLoss_W1 = np.dot(dLoss_Z1.T, X)
    return dLoss_W1, dLoss_W2, dLoss_W3

def crossEntropyLoss(y_train, Yh):
    # loss = []
    loss = (1. / len(y_train)) * (-np.dot(y_train, np.log(Yh).T) - np.dot(1 - y_train, np.log(1 - Yh).T))
    return loss

def SGD(W1, W2, W3, traindata, testdata, T, B,lambdaval):
    lossTrain = []
    lossTest = []
    accuracytrain = []
    accuracytest = []
    lambda_dict={}
    p = 0
    temp = -1
    for t in range(1, T):
        print("iteration", t)
        print("lambdaval",lambdaval)
        lr = lambdaval / t
        X, y_train = getBatchdata(traindata, B)
        Z1, Z2, Z3, A1, A2, A3 = forward(W1, W2, W3, X)
        A3 = np.asarray(A3)
        Yh = A3
        dLoss_W1, dLoss_W2, dLoss_W3 = backward(Z3, Z2, Z1, A1, A2, X, y_train, Yh[0])
        if (np.isnan(Yh[0][0])):
            lambda_dict['lastValidationerror'] = None
            lambda_dict['W1'] = None
            lambda_dict['W2'] = None
            lambda_dict['W3'] = None
            lambda_dict['lambdaval'] = lr
            #print(lambda_dict)
            return lambda_dict
        W1 = W1 - (lr * dLoss_W1)
        W2 = W2 - (lr * dLoss_W2)
        W3 = W3 - (lr * dLoss_W3)
        if (t % 250 == 0):
            #findind output for validation data after each 250th iteration of training data
            p = p + 1
            Xtest, y_test = getBatchdata(testdata, B)
            z1, z2, z3, a1, a2, a3 = forward(W1, W2, W3, Xtest)
            a3 = np.asarray(a3)
            yh = a3
            currentLoss = crossEntropyLoss(y_test, yh[0])
            if ((p >= 5) & (len(lossTest) > 0)):
                temp = temp + 1
                if (currentLoss > np.amax(lossTest[temp:p])):
                    #plotAccuracyGraph(t, accuracytrain, accuracytest)
                    #A dictionary to store last validation error ,lambda and weights
                    lambda_dict['lastValidationerror'] = lossTrain[p-2]
                    lambda_dict['W1'] = W1+(lr * dLoss_W1)
                    lambda_dict['W2'] = W2 + (lr * dLoss_W2)
                    lambda_dict['W3']=W3 + (lr * dLoss_W3)
                    lambda_dict['lambdaval']=lr
                    return lambda_dict
            lossTest.append(currentLoss)
            accuracytest.append(accuracy(y_test, yh[0]))
            lossTrain.append(crossEntropyLoss(y_train, Yh[0]))
            accuracytrain.append(accuracy(y_train, Yh[0]))

def plotAccuracyGraph(t, accuracytrain, accuracytest):
    print("accuracy of train data:", accuracytrain)
    print("accuracy of train data:", accuracytest)
    epoch = [i for i in range(250, t - 1, 250)]
    plt.plot(epoch, np.asarray(accuracytrain), 'r--')
    plt.plot(epoch, np.asarray(accuracytest), 'b-')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(in %)')
    plt.savefig('accuracygraph.png')
    plt.show();
def plotLambdagraph(lambdaval,lastValidationError):
    plt.plot(lambdaval, np.asarray(lastValidationError), 'r-')
    plt.xlabel('Learning rate(lambda/t)')
    plt.ylabel('Final validation error.')
    plt.savefig('lambdagraph.png')
    plt.show();
def accuracy(y_train, yh):
    acc = 0
    for i in range(0, len(y_train)):
        if (yh[i] > 0.5):
            y = 1
        else:
            y = 0
        if (y_train[i] == y):
            acc = acc + 1
    return (acc / len(y_train)) * 100

def getBatchdata(traindata, B):
    train = shuffle(traindata)
    train_X = train.iloc[:, 1:3].values
    y_traindata = train.iloc[:, 3].values
    X = train_X[0:B]
    y_train = y_traindata[0:B]
    return np.asarray(X), np.asarray(y_train)

if __name__ == "__main__":
    df = pd.read_csv("BIG_DWH_Training.csv", sep=',',
                     names=["index", "Height", "weight", "gender"])
    femaleSet = df.loc[df['gender'] == -1]
    maleSet = df.loc[df['gender'] == 1]
    dim = [2, 5, 5, 1]
    df = shuffle(df)
    scaler = preprocessing.StandardScaler().fit(df.iloc[:, 1:3].values)
    Xtrain = scaler.transform(df.iloc[:, 1:3].values)
    df['Height'] = Xtrain[:, 0]
    df['weight'] = Xtrain[:, 1]
    train_len = int(len(df) * .95)
    test_len = int(len(df) * .05)
    train = df[0:train_len]
    test = df[train_len:train_len + test_len]
    lastValidationError=[]
    W_1 = []
    W_2 = []
    W_3 = []
    lambdaval=[]
    #error=[]
    lambda_dict = {}
    for i in range(1, 11):
        W1 = initializeWeight(0)
        W2 = initializeWeight(1)
        W3 = initializeWeight(2)
        lambda_dict = SGD(W1, W2, W3, train, test, 100000, 50, i)
        if lambda_dict['W1'] is not None:
            W_1.append(lambda_dict['W1'])
            W_2.append(lambda_dict['W2'])
            W_3.append(lambda_dict['W3'])
            lastValidationError.append(lambda_dict['lastValidationerror'])
            lambdaval.append(lambda_dict['lambdaval'])
    index=0
    plotLambdagraph(lambdaval,lastValidationError)
    #print("lasttt",lastValidationError)
    minimum = lastValidationError[0]
    for i in range(1, len(lastValidationError)):
        if (lastValidationError[i] < minimum):
            minimum = lastValidationError[i]
            index = i
    #print( lambdaval)
    print("Best learning rate is:", lambdaval[index],"index:",index)

    print("min error", lastValidationError[index])
    W1=W_1[index]
    W2=W_2[index]
    W3=W_3[index]
    # finally calculate accuracy of test set
    dfTest = pd.read_csv("DWH_Test.csv", sep=',',
                         names=["index", "height", "weight", "gender", "distance"])
    Xtest = dfTest.iloc[:, 1:3].values
    Ytest = dfTest.iloc[:, 3].values
    scaler = preprocessing.StandardScaler().fit(Xtest)
    testdata = scaler.transform(Xtest)
    Z1, Z2, Z3, A1, A2, A3 = forward(W1, W2, W3, testdata)
    A3 = np.asarray(A3)
    Yh = A3
    acc = accuracy(Ytest, Yh[0])
    print("Accuracy of the test set is",acc,"%")



