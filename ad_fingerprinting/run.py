import recognize
import numpy as np
import os
import sys 
from scipy.io.wavfile import read
from sklearn import svm
import csv
import pickle
import random

model_save_file = 'learned_model.pkl'

def test(x_test):
    with open(model_save_file, 'rb') as file:  
        learned_model = pickle.load(file)
        prediction = learned_model.predict(x_test)
        n_test_samples = x_test.shape[0]
        # print("Accuracy = ", 1 - np.sum(abs(prediction - y_test))/n_test_samples)

    return prediction    


feature = []

input_file = 'niviea.csv'        # path to csv file
rows = [] 
with open(input_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader: 
        rows.append(row)
    print("Total no. of rows: %d"%(csvreader.line_num))

rows = np.array(rows)
rows = rows.astype(np.float64)
# np.random.shuffle(rows)

n_samples, columns = rows.shape
n_feature = columns-1
n_test_samples = int(n_samples/2)
n_training_samples = n_samples - n_test_samples

x_train = rows[0:n_training_samples, ]
# y_train = rows[0:n_training_samples, columns-1]
x_test = rows[n_training_samples:n_samples, :]
# y_test = rows[n_training_samples:n_samples, columns-1]

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train = (x_train - mean)/std
x_test = (x_test - mean)/std

# print("Number of Positive samples = ", np.sum(y_train))
# print("Number of Negative samples = ", n_samples - np.sum(y_train))
filepath = 'niviea.mp3'
pred = test(x_test)
if np.sum(pred) > n_samples/4 :
    print("AD DETECTED!")
    recognize.recognize(filepath)