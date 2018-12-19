#!/usr/bin/env python3
# -*- coding: utf-8 -*-






net = initWeight()
training(X_train, Y_train, 1000)

print("Hello REPL")




df = pd.DataFrame([list(i) for i in zip(*time_series_dataset)])

X_train = df.iloc[0:90, :4]
Y_train = df.iloc[0:90:, 4]
X_test = df.iloc[71:80, :4]
Y_test = df.iloc[71:80, 4]

max_dataset = 1292.0
min_dataset = 538.0

backpp = BackpropagationNN(4, 3, 1, 0.4)
backpp.initWeight()
backpp.training(X_train, Y_train, max_dataset, min_dataset, 100)
backpp.data_testing(X_test, Y_test, max_dataset, min_dataset)
