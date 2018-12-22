import numpy as np
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import csv
import pickle
import random
import pdb
import time

cache_size = 200
kernel = 'linear'
max_iteration = -1
model_save_file = 'learned_model.pkl'


def train(x_train, y_train):
    start_time = time.time()
    # parameters = {'C':[0.01, 0.1, 1.0, 10.0]}
    weak1 = svm.SVC(C=1, gamma='scale', class_weight='balanced', cache_size=cache_size, kernel=kernel, max_iter=max_iteration, probability=True)
    # weak1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=10, learning_rate=1.0, algorithm='SAMME.R')
    clf = AdaBoostClassifier(weak1, n_estimators=10, learning_rate=1.0, algorithm='SAMME.R')
    # clf = GridSearchCV(svc, parameters)
    clf.fit(x_train, y_train)
    end_time = time.time()
    print("Training ended, Time (in mins) = ", (end_time - start_time)/60.0)
    pickle.dump(clf, open(model_save_file, 'wb'))

def test(x_test, y_test):
    with open(model_save_file, 'rb') as file:  
        learned_model = pickle.load(file)
        prediction = learned_model.predict(x_test)
        n_test_samples = x_test.shape[0]
        print("Accuracy = ", 1 - np.sum(abs(prediction - y_test))/n_test_samples)


if __name__ == "__main__":
    input_file = 'combined_labels_Sheet2.csv'        # path to csv file
    rows = [] 
    with open(input_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader: 
            rows.append(row)
        print("Total no. of rows: %d"%(csvreader.line_num))

    rows = np.array(rows)
    rows = rows.astype(np.float64)
    np.random.shuffle(rows)

    n_samples, columns = rows.shape
    n_feature = columns-1
    x_train, x_test, y_train, y_test = train_test_split(rows[:,0:n_feature], rows[:,n_feature], test_size=0.20)

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean)/std
    x_test = (x_test - mean)/std

    print("Number of Positive samples = ", np.sum(y_train))
    print("Number of Negative samples = ", n_samples - np.sum(y_train))
    train(x_train, y_train)
    test(x_test, y_test)