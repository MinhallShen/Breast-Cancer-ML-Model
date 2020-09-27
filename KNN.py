# Created by: Minhall Shen
# This is a machine learning model that utilizes the k-nearest neighbours algorithm that takes the data of breast cancer
# patients and determines whether the cancer is recurring or non-recurring. The dataset was taken from the UCI Machine
# Learning Repository (archive.ics.uci.edu/ml/datasets/Breast+Cancer) and obtained from the University Medical
# Centre, Institute of Oncology, Ljubljana, Yugoslavia (see breast-cancer.names)

import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import csv

# create dataframe from data
file = open('breast-cancer.data')
reader = csv.reader(file)
num_lines = len(list(reader)) - 1  # exclude top row of headers

df = pd.read_csv('breast-cancer.data')

l = preprocessing.LabelEncoder()

# turn attributes into numerical values
cla = l.fit_transform(list(df['Class']))
age = l.fit_transform(list(df['age']))
menopause = l.fit_transform(list(df['menopause']))
tumor_size = l.fit_transform(list(df['tumor size']))
inv_nodes = l.fit_transform(list(df['inv-nodes']))
node_caps = l.fit_transform(list(df['node-caps']))
deg_malig = l.fit_transform(list(df['deg-malig']))
breast = l.fit_transform(list(df['breast']))
breast_quad = l.fit_transform(list(df['breast-quad']))
irradiat = l.fit_transform((list(df['irradiat'])))

# create a nested array; array of clients and their attributes
x = list(zip(age, menopause, tumor_size, inv_nodes, node_caps, deg_malig, breast, breast_quad, irradiat))
# array of classes (non-recurring or recurring); the attribute we want to predict
y = list(cla)

size = 0.1
# split into two random sets, one for training and one for testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=size)

# value of k for k nearest neighbors
k_value = 0
max_accuracy = 0

# find optimal k value and its accuracy (k must be odd)
for i in range(0,int((num_lines-(0.1*num_lines))/2)):
    temp_model = KNeighborsClassifier(n_neighbors=2*i+1)
    temp_model.fit(x_train, y_train)
    temp_accuracy = temp_model.score(x_test, y_test)
    if temp_accuracy > max_accuracy:
        max_accuracy = temp_accuracy
        k_value = 2*i + 1

# apply k value
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(x_train, y_train)

# array of predictions made by model
predicted_class = model.predict(x_test)
names = ["no recurrence events", "recurrence events"]

# print out all the patients and the model's predictions and indicate whether or not the model was correct
for i in range(len(predicted_class)):
    if names[predicted_class[i]] == names[y_test[i]]:
        print(str(i+1) + ": " + "Data: " + str(x_test[i]) + " Predicted: " + names[predicted_class[i]] + " Actual: " +
              names[y_test[i]] + "; Correct Prediction")
    else:
        print(str(i+1) + ": " + "Data: " + str(x_test[i]) + "Predicted: " + names[predicted_class[i]] + " Actual: " +
              names[y_test[i]] + "; Incorrect Prediction")

print("k = " + str(k_value))
print("Accuracy: " + str(int(np.round(100*max_accuracy))) + "%")

# write the data to a txt file
write = raw_input("Save this data to a txt file? Type 'yes' if you would and anything else otherwise: ")

if write == "yes":
    filename = raw_input("Please enter the file's name: ")
    f = open(filename + ".txt", "w")
    for i in range(len(predicted_class)):
        f = open(filename + ".txt", "a")
        if names[predicted_class[i]] == names[y_test[i]]:
            f.write(str(i+1) + ": " + "Data: " + str(x_test[i]) + " Predicted: " + names[predicted_class[i]] +
                    " Actual: " + names[y_test[i]] + "; Correct Prediction\n")
            f.close()
        else:
            f.write(str(i+1) + ": " + "Data: " + str(x_test[i]) + "Predicted: " + names[predicted_class[i]] +
                    " Actual: " + names[y_test[i]] + "; Incorrect Prediction\n")
            f.close()
    g = open(filename + ".txt", "a")
    g.write("Accuracy: " + str(int(np.round(100*max_accuracy))) + "%\n")
    g.close()
