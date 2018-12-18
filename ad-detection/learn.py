import numpy as np
from sklearn import svm
import csv
import pickle
import random
import pdb

alpha = 1.0    # regularization parameter
gamma = 'auto'
cache_size = 200
kernel = 'rbf'
max_iteration = -1
model_save_file = 'learned_model.pkl'

def train(x_train, y_train):
    clf = svm.SVC(C=1/alpha, gamma=gamma, class_weight='balanced', cache_size=cache_size, kernel=kernel, max_iter=max_iteration, probability=True)
    clf.fit(x_train, y_train)
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
    n_test_samples = int(n_samples/5)
    n_training_samples = n_samples - n_test_samples

    x_train = rows[0:n_training_samples, :-1]
    y_train = rows[0:n_training_samples, columns-1]
    x_test = rows[n_training_samples:n_samples, :-1]
    y_test = rows[n_training_samples:n_samples, columns-1]

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train = (x_train - mean)/std
    x_test = (x_test - mean)/std

    print("Number of Positive samples = ", np.sum(y_train))
    print("Number of Negative samples = ", n_samples - np.sum(y_train))
    train(x_train, y_train)
    test(x_test, y_test)
