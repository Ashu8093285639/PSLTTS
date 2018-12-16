#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from random import random as rand

dataset = pd.read_csv('wheat-seeds-normalized.csv',header=None)
dataset = dataset.drop(8, axis=1)

class Neuron:
    
    ## weight == array of weight
    def __init__(self, weight):
        self.weight = weight
        self.output = list()
        self.delta = list()


def convertClassToTarget(y):    
    val = np.binary_repr(y, width=2)
    target = list()
    for idt, t in enumerate((val + "")):
        if t == "0": 
            t = "-1" ##bipolar
        target.append(int(t))
    return target



X_train, X_test, Y_train, Y_test = ms.train_test_split(dataset.iloc[:, 0:-1], dataset.iloc[:, -1].apply(convertClassToTarget), test_size=0.30)
X_list = X_train.values.tolist()
Y_list = Y_train.values.tolist()
X_test_list = X_test.values.tolist()
Y_test_list = Y_test.values.tolist()

    
BIAS_ID = 0
THRESHOLD = 0.4
INPUT_LAYER = len(X_train.columns)
HIDDEN_LAYER = 2
OUTPUT_LAYER = 2


LEARNING_RATE = 0.4


def summingFunction(last_weight, training_data):
    bias = last_weight[-1]
    y_in = 0
    for idx in range(0, len(last_weight) - 1):
        y_in += last_weight[idx] * training_data[idx]        
    return bias + y_in

def activation(y_in):
    return ( 2 / (1 + np.exp(-1 * y_in))) - 1

def thresholdFunction(y):
    if y > THRESHOLD:
        return 1
    elif y < THRESHOLD * -1:
        return -1
    else:
        return 0

def derivativeFunction(y):
    return 0.5 * (1+y) * (1-y)
        
##def feedForward(layer1, layer2):
        
def initWeight():
    layer = list()
    input_to_hidden = [ Neuron([rand() for i in range(INPUT_LAYER + 1)]) for j in range (HIDDEN_LAYER) ]
    layer.append(input_to_hidden)
    hidden_to_output = [ Neuron([rand() for i in range(HIDDEN_LAYER + 1)]) for j in range (OUTPUT_LAYER) ]
    layer.append(hidden_to_output)
    return layer

def feedForward(row):
    processed_layer = row
    for layer in net:
        ## print(layer)
        new_inputs = list() ##
        for neuron in layer:
            ##print(neuron)
            y_in = summingFunction(neuron.weight, processed_layer)
            neuron.output = activation(y_in)
            new_inputs.append(neuron.output)
        processed_layer = new_inputs
    return processed_layer ## nilai y1, y2, .... yx



def backPpError(target):
    for id_layer in reversed(range(len(net))):
        layer = net[id_layer]
        ##print(id_layer)
        error_factor = list()
        if id_layer == len(net) - 1: # error_factor dari output
            for weight_id in range(len(layer)):
                neuron = layer[weight_id]
                error_factor.append(target[weight_id] - neuron.output)
        else:
            for weight_id in range(len(layer)):
                error_factor_sum = 0.0
                prev_layer = net[id_layer - 1]
                for neuron in prev_layer:
                    error_factor_sum += (neuron.weight[weight_id] * neuron.delta)
                error_factor.append(error_factor_sum)
                
        for weight_id in range(len(layer)):
            neuron = layer[weight_id]
            neuron.delta = error_factor[weight_id] * derivativeFunction(neuron.output)
   
def updateWeight(row):
    for i in range(len(net)):
        inputs = list()
        if i == 0:
            inputs = row
        else:
            prev_net = net[i - 1]
            inputs = [neuron.output for neuron in prev_net]
        ##print(inputs)
        for neuron in net[i]:
            neuron.weight[-1] += LEARNING_RATE * neuron.delta
            for weight_id in range(0, len(inputs)):
                neuron.weight[weight_id] += LEARNING_RATE * neuron.delta * inputs[weight_id]

            
def training(X_train, Y_train, total_epoch):
    X_list = X_train.values.tolist()
    Y_list = [[x] for x in Y_train.values.tolist()]

    for epoch in range(total_epoch):        
        mse = 0
        for idx, data_row in enumerate(X_list):
            ff_output = feedForward(X_list[idx])
            target = Y_list[idx]
            se = 0
            for idy, y in enumerate(ff_output):
                se += np.square(target[idy] - ff_output[idy])
            mse += se
            backPpError(Y_list[idx])
            updateWeight(X_list[idx])        
        print('epoch %s, MSE %s' % (epoch, mse/len(X_list)))
        printNetworkWeight()
        print()
    

def predict(row):
    output = feedForward(row)
    ##print(output)
    classifier = list()
    for y in output:
        classifier.append(thresholdFunction(y))
    return classifier

def printNetworkWeight():
    input_to_hidden = net[0]
    hidden_to_output = net[1]
    
    print('Weight summary')
    print('Input to Hidden')
    for idx, neuron in enumerate(input_to_hidden):
        strn = ''
        for idy, weight in enumerate(neuron.weight):
            strn += 'v' + str(idx) + str(idy) + '  '
        print('%s : %s' % (strn, neuron.weight))
    
    
    print('Hidden to Output')
    for idx, neuron in enumerate(hidden_to_output):
        strn = ''
        for idy, weight in enumerate(neuron.weight):
            strn += 'w' + str(idx) + str(idy) + '  '
        print('%s : %s' % (strn, neuron.weight))
    


net = initWeight()
training(X_train, Y_train, 1000)
predict([-1.46322, -1.64575, -0.26232, -1.64657, -1.2485700000000002, 0.8478549999999999, -1.2494299999999998])
print('Validasi dengan data training')
count_val = 0
right_val = 0
for idx, row in enumerate(X_list):
    output = predict(row)
    target = Y_list[idx]
    if output[0] == target[0] and output[1] == target[1]:
        right_val += 1
    else:
        print('%s %s || %s %s' % (output[0], output[1], target[0], target[1]))
#        print(row)
    count_val += 1

print('Akurasi %s' % (right_val/count_val *100))
 
print('Evaluasi dengan data testing')
count = 0
right = 0
for idx, row in enumerate(X_test_list):
    output = predict(row)
    target = Y_test_list[idx]
    if output[0] == target[0] and output[1] == target[1]:
        right += 1
    else:
        print('%s %s || %s %s' % (output[0], output[1], target[0], target[1]))
#        print(row)
    count += 1

print('Akurasi %s' % (right/count *100))
