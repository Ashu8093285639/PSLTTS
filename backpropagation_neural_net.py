import pandas as pd
import numpy as np
import random as rd


class Neuron:
    # weight == array of weight
    def __init__(self, weight):
        self.weight = weight
        self.output = list()
        self.delta = list()


INPUT_LAYER = 5
HIDDEN_LAYER = 3
OUTPUT_LAYER = 1
LEARNING_RATE = 0.5

# total bobot = 22
# 0,1,2,3,4,5


def summingFunction(last_weight, training_data):
    bias = last_weight[-1]  # bias on last index
    y_in = 0
    for idx in range(0, len(last_weight) - 1):
        y_in += last_weight[idx] * training_data[idx]
    return bias + y_in


def activation(y_in):
    return 1.0 / (1.0 + np.exp(-1 * y_in))


def derivativeFunction(y):
    return y * (1.0 - y)


def initWeight():
    layer = list()
    input_to_hidden = [
        Neuron([rd.random() for i in range(INPUT_LAYER + 1)]) for j in range
        (HIDDEN_LAYER)]
    layer.append(input_to_hidden)
    hidden_to_output = [
        Neuron([rd.random() for i in range(HIDDEN_LAYER + 1)]) for j in range
        (OUTPUT_LAYER)]
    layer.append(hidden_to_output)
    return layer


def initWeightPSO(partikel):
    layer = list()
    partikel_dimens_idx = 0
    input_to_hidden = list()

    for i in range(HIDDEN_LAYER):
        w = list()
        for j in range(INPUT_LAYER + 1):
            w.append(partikel[partikel_dimens_idx])
            partikel_dimens_idx += 1
        input_to_hidden.append(Neuron(w))
    layer.append(input_to_hidden)

    hidden_to_output = list()
    for i in range(OUTPUT_LAYER):
        w = list()
        for j in range(HIDDEN_LAYER + 1):
            w.append(partikel[partikel_dimens_idx])
            partikel_dimens_idx += 1
        hidden_to_output.append(Neuron(w))
    layer.append(hidden_to_output)
    return layer


def feedForward(net, row):
    processed_layer = row
    for layer in net:
        # print(layer)
        new_inputs = list()
        for neuron in layer:

            # print(neuron)
            y_in = summingFunction(neuron.weight, processed_layer)
            neuron.output = activation(y_in)
            new_inputs.append(neuron.output)
        processed_layer = new_inputs
    return processed_layer  # nilai y1, y2, .... yx


def backPpError(target):
    for id_layer in reversed(range(len(net))):
        layer = net[id_layer]
        # print(id_layer)
        error_factor = list()
        if id_layer == len(net) - 1:  # error_factor dari output
            for weight_id in range(len(layer)):
                neuron = layer[weight_id]
                error_factor.append(target[weight_id] - neuron.output)
        else:
            for weight_id in range(len(layer)):
                error_factor_sum = 0.0
                prev_layer = net[id_layer - 1]
                for neuron in prev_layer:
                    error_factor_sum += neuron.weight[weight_id] * neuron.delta
                error_factor.append(error_factor_sum)

        for weight_id in range(len(layer)):
            neuron = layer[weight_id]
            neuron.delta = error_factor[weight_id] * derivativeFunction(
                neuron.output)


def updateWeight(row):
    for i in range(len(net)):
        inputs = list()
        if i == 0:
            inputs = row
        else:
            prev_net = net[i - 1]
            inputs = [neuron.output for neuron in prev_net]
        # print(inputs)
        for neuron in net[i]:
            neuron.weight[-1] += LEARNING_RATE * neuron.delta
            for weight_id in range(0, len(inputs)):
                # print(weight_id)
                neuron.weight[weight_id] += LEARNING_RATE * (
                    neuron.delta - inputs[weight_id])


def training(net, X_train, Y_train, total_epoch):
    X_list = X_train.values.tolist()
    Y_list = [[x] for x in Y_train.values.tolist()]
    fitness = 0
    for epoch in range(total_epoch):
        mape = 0
        actual_value = list()
        for idx, data_row in enumerate(X_list):
            feedForward(net, X_list[idx])
            actual_value.append(un_normalized(Y_train[idx]))
            backPpError(Y_list[idx])
            updateWeight(X_list[idx])
        for x in range(len(actual_value)):
            forecasted_value = un_normalized(predict(net, X_list[x])[0])
            # print('Test: ', actual_value[x], " ", forecasted_value)
            mape += np.abs(
                (actual_value[x] - forecasted_value) / actual_value[x])
        mape = mape * 100 / len(actual_value)
        fitness = 100 / (100 + mape)
        print('epoch %s, MAPE %s, FITNESS %s' % (epoch, mape, fitness))
        # printNetworkWeight()
        #print()
    return fitness


def predict(net, row):
    output = feedForward(net, row)
    return output


def un_normalized(value):
    return value * (max(dataset_value) - 
        min(dataset_value)) + min(dataset_value)


df = pd.DataFrame([list(i) for i in zip(*time_series_dataset)])

X_train = df.iloc[:, :5]
Y_train = df.iloc[:, 5]

net = initWeight()
training(net, X_train, Y_train, 100)

