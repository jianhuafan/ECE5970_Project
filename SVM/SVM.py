from sklearn import svm
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import csv
import numpy as np
import sys
import matplotlib.pyplot as plt
# n_samples, n_features = 551, 1
# clif = SVR(C=1.0, epsilon=0.2)
# myarray = np.asarray(loss)
# print(myarray)
# myarray = myarray.reshape(-1,1)
# clif.fit(myarray, accuracy)
# y = clif.predict(newarray)
# y = y.tolist()
def feature4Prediction(name, X, y):
    n_samples = len(X)
    n_features = len(X[0])
    clif = SVR()
    clif.fit(X, y)
    with open('input_test{}.csv'.format(name), 'rb') as test_input_file:
        X_test = list(csv.reader(test_input_file))
    X_test = np.array(X_test)
    y_hat = clif.predict(X_test)
    y_hat = y_hat.tolist()
    with open('target_test{}.csv'.format(name), 'rb') as test_output_file:
        total_y_test = list(csv.reader(test_output_file))
    y_test = np.array(total_y_test)
    y_test = y_test[:,1:]
    return AccuracyOfSVR(y_test, y_hat)

def crossValidation(name, X, y):
    kernellist = ['rbf', 'sigmoid','linear','poly']
    degreelist = [2, 3, 4, 5, 6]
    gammalist = np.arange(0.001,0.5,0.001)
    for k in kernellist:
        clif = SVR(kernel=k)
        clif.fit(X, y)
        with open('input_test{}.csv'.format(name), 'rb') as test_input_file:
            X_test = list(csv.reader(test_input_file))
        X_test = np.array(X_test)
        y_hat = clif.predict(X_test)
        y_hat = y_hat.tolist()
        with open('target_test{}.csv'.format(name), 'rb') as test_output_file:
            total_y_test = list(csv.reader(test_output_file))
        y_test = np.array(total_y_test)
        y_test = y_test[:,1:]
        print(k, AccuracyOfSVR(y_test, y_hat), meanDifference(y_test, y_hat))

def crossValidationOfSVC(name, X, y):
    kernellist = ['rbf', 'sigmoid','linear','poly']
    degreelist = [2, 3, 4, 5]
    gammalist = []
    for k in degreelist:
        clif = SVC(kernel='poly',degree=k)
        clif.fit(X, y)
        with open('input_test{}.csv'.format(name), 'rb') as test_input_file:
            X_test = list(csv.reader(test_input_file))
        X_test = np.array(X_test)
        y_hat = clif.predict(X_test)
        y_hat = y_hat.tolist()
        with open('target_test{}.csv'.format(name), 'rb') as test_output_file:
            total_y_test = list(csv.reader(test_output_file))
        y_test = np.array(total_y_test)
        y_test = y_test[:,1:]
        y_test = transform(y_test)
        print(k, AccuracyOfSVC(y_test, y_hat))



def transform(original):
    y_SVC = []
    for row in range(len(original)):
        if original[row, 0] == '1.0':
            y_SVC += [0]
        elif original[row, 1] == '1.0':
            y_SVC += [1]
        elif original[row, 2] == '1.0':
            y_SVC += [2]
    return y_SVC

def B_SVC(name, X, y):
    n_samples = len(X)
    n_features = len(X[0])
    clif = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True))
    y = label_binarize(y, classes=[0,1,2])
    n_classes = y.shape[1]
    clif.fit(X, y)
    with open('input_test{}.csv'.format(name), 'rb') as test_input_file:
        X_test = list(csv.reader(test_input_file))
    X_test = np.array(X_test,dtype=float)
    y_hat = clif.predict(X_test)
    y_hat = y_hat.tolist()
    with open('target_test{}.csv'.format(name), 'rb') as test_output_file:
        total_y_test = list(csv.reader(test_output_file))
    y_test = np.array(total_y_test)
    y_test = y_test[:,1:]
    # not_y_test = y_test
    y_test = transform(y_test)
    y_test = label_binarize(y_test, classes=[0,1,2])
    y_score = clif.decision_function(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    variable = 0
    lw = 2
    plt.plot(fpr[variable], tpr[variable], color='darkorange',
         lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc[variable])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig("{}.png".format(variable))
    plt.show()
    raw_input("press enter to close plots")
    plt.close("all")

def B_SVR(name, X, y):
    n_samples = len(X)
    n_features = len(X[0])
    clif = SVR(C = 1.0, epsilon = 0.2)
    clif.fit(X, y)
    with open('input_test{}.csv'.format(name), 'rb') as test_input_file:
        X_test = list(csv.reader(test_input_file))
    X_test = np.array(X_test)
    y_hat = clif.predict(X_test)
    y_hat = y_hat.tolist()
    with open('target_test{}.csv'.format(name), 'rb') as test_output_file:
        total_y_test = list(csv.reader(test_output_file))
    y_test = np.array(total_y_test)
    y_test = y_test[:,1:]
    return AccuracyOfSVR(y_test, y_hat), meanDifference(y_test, y_hat)

def meanDifference(y_test, y_hat):
    difference = []
    zipped = zip(y_test, y_hat)
    for i, j in zipped:
        i = np.asscalar(i)
        i = float(i)
        difference.append(abs(i - j))
    difference = np.array(difference)
    return np.mean(difference)

def AccuracyOfSVC(y_test, y_hat):
    correct = 0
    # print(len(y_test), len(y_hat))
    # print(y_test)
    # print(type(y_test[0]), type(y_hat[0]))

    zipped = zip(y_test, y_hat)
    for i, j in zipped:
        # temp = ''
        # for k in i:
        #     temp += str(k)
        if i == j:
            correct += 1
    accuracy = float(correct) / len(y_test)
    return accuracy

def AccuracyOfSVR(y_test, y_hat):
    correct = 0
    # print(len(y_test), len(y_hat))
    # print(y_test)
    # print(type(y_test[0]), type(y_hat[0]))
    zipped = zip(y_test, y_hat)
    for i, j in zipped:
        i = np.asscalar(i)
        if 0.2 * float(i) >= abs(float(i) - j):
            correct += 1
    accuracy = float(correct) / len(y_test)
    return accuracy

def getVariable():
    name = sys.argv[1]
    with open('input_train{}.csv'.format(name), 'rb') as train_input_file:
        X = list(csv.reader(train_input_file))
    with open('target_train{}.csv'.format(name), 'rb') as train_output_file:
        total_y = list(csv.reader(train_output_file))
    X = np.array(X, dtype=float)
    y = np.array(total_y)
    if name != '5':
        y = y[:,1:]
    if name == '1':
        y_SVC = transform(y)
        #crossValidationOfSVC(name, X, y_SVC)
        accuracy = B_SVC(name, X, y_SVC)
        print(accuracy)
    elif name == '5':
        accuracy = feature4Prediction(name, X, y)
        print(accuracy)
    else:
        crossValidation(name, X, y)
        #accuracy, meanDifference = B_SVR(name, X, y)
        #print(accuracy, meanDifference)




if __name__ == "__main__":
    getVariable()

