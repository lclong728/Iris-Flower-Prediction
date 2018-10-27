# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:34:23 2018

@author: lenovo
"""

import sklearn.svm as svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score as accuracy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# loading data
def load_iris_data():
    name = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    return pd.read_csv("iris_predict_data.csv", header=None, names=name)


# split out train and test data
def split_data(all_data):
    return model_selection.train_test_split(all_data, test_size=0.2, random_state=7)


# split train and test dataFrame
def iris_data_info(train_data, test_data):
    print("train shape:   ")
    print(train_data.shape)
    print("test shape:    ")
    print(test_data.shape)
    print("Unique classes with count : ")
    print(pd.value_counts(train_data['class']))
    print("data description : ")
    print(train_data.describe())
    print("train missing value check: ")
    print(train_data.info())
    return


# data visualization for analysis
def visualization_train_data(train_data):
    plt.hist(train_data['class'])
    plt.show(block=True)
    sns.distplot(train_data['sepal_length'], kde=True, bins=40)
    plt.show(block=True)
    sns.pairplot(train_data, hue='class', palette='OrRd')
    plt.show()
    return


def svm_model(train_data_set, train_target_set):
    svc = svm.SVC()
    svc.fit(train_data_set, train_target_set)
    return svc


if __name__ == '__main__':
    data = load_iris_data()
    train, test = split_data(data)
    # iris_data_info(train, test)

    data_train = train.iloc[:, :-1]
    target_train = train.iloc[:, -1]
    data_test = test.iloc[:, :-1]
    target_test = test.iloc[:, -1]

    prediction = svm_model(data_train, target_train).predict(data_test)
    print("accuracy = " + str(accuracy(target_test, prediction)))