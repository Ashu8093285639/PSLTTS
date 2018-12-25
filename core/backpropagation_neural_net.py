from numpy import abs
from numpy import exp
from random import random


def un_normalized(value, min, max):
    return value * (max - min) + min


class Neuron:
    # weight == array of weight
    def __init__(self, weight):
        self.weight = weight
        self.output = list()
        self.delta = list()


class BackpropagationNN:

    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate):
        self.INPUT_LAYER = input_layer
        self.HIDDEN_LAYER = hidden_layer
        self.OUTPUT_LAYER = output_layer
        self.LEARNING_RATE = learning_rate

    def __str__(self):
        str_buff = ""
        str_buff += "Input to Hidden\n"
        for neuron in self.net[0]:
            str_buff += str(neuron.weight) + "\n"
        str_buff += "Hidden to Output\n"
        for neuron in self.net[1]:
            str_buff += str(neuron.weight) + "\n"
        return str_buff

    def summingFunction(self, last_weight, training_data):
        bias = last_weight[-1]  # bias on last index
        y_in = 0
        for idx in range(0, len(last_weight) - 1):
            y_in += last_weight[idx] * training_data[idx]
        return bias + y_in

    def activation(self, y_in):
        return 1.0 / (1.0 + exp(-1 * y_in))

    def derivativeFunction(self, y):
        return y * (1.0 - y)

    def initWeight(self):
        layer = list()
        input_to_hidden = [
            Neuron([random() for i in range(
                self.INPUT_LAYER + 1)]) for j in range(self.HIDDEN_LAYER)]
        layer.append(input_to_hidden)
        hidden_to_output = [
            Neuron([random() for i in range(
                self.HIDDEN_LAYER + 1)]) for j in range(self.OUTPUT_LAYER)]
        layer.append(hidden_to_output)
        self.net = layer

    def feedForward(self, row):
        processed_layer = row
        for layer in self.net:
            # print(layer)
            new_inputs = list()
            for neuron in layer:

                # print(neuron)
                y_in = self.summingFunction(neuron.weight, processed_layer)
                neuron.output = self.activation(y_in)
                new_inputs.append(neuron.output)
            processed_layer = new_inputs
        return processed_layer  # nilai y1, y2, .... yx

    def backPpError(self, target):
        for id_layer in reversed(range(len(self.net))):
            layer = self.net[id_layer]
            # print(id_layer)
            error_factor = list()
            if id_layer == len(self.net) - 1:  # error_factor dari output
                for weight_id in range(len(layer)):
                    neuron = layer[weight_id]
                    error_factor.append(target[weight_id] - neuron.output)
            else:
                for weight_id in range(len(layer)):
                    error_factor_sum = 0.0
                    prev_layer = self.net[id_layer - 1]
                    for neuron in prev_layer:
                        error_factor_sum += neuron.weight[weight_id] * neuron.delta
                    error_factor.append(error_factor_sum)

            for weight_id in range(len(layer)):
                neuron = layer[weight_id]
                neuron.delta = error_factor[weight_id] * self.derivativeFunction(neuron.output)

    def updateWeight(self, row):
        for i in range(len(self.net)):
            inputs = list()
            if i == 0:
                inputs = row
            else:
                prev_net = self.net[i - 1]
                inputs = [neuron.output for neuron in prev_net]
            # print(inputs)
            for neuron in self.net[i]:
                neuron.weight[-1] += self.LEARNING_RATE * neuron.delta
                for weight_id in range(len(neuron.weight) - 1):
                    neuron.weight[weight_id] += self.LEARNING_RATE * (
                        neuron.delta - inputs[weight_id])

    def training(self, X_train, Y_train, max_value, min_value, total_epoch):
        # print(self.net[0][0].weight)
        X_list = X_train.values.tolist()
        Y_list = [[x] for x in Y_train.values.tolist()]
        for epoch in range(total_epoch):
            mape = 0
            for idx, data_row in enumerate(Y_list):
                self.feedForward(X_list[idx])
                self.backPpError(Y_list[idx])
                self.updateWeight(X_list[idx])
            for x in range(len(Y_list)):
                actual_value = un_normalized(
                    Y_list[x][0], max_value, min_value)
                forecasted_value = un_normalized(
                    self.predict(X_list[x])[0], max_value, min_value)
                # print('Test: ', actual_value, " ", forecasted_value)
                mape += abs(
                    (actual_value - forecasted_value) / actual_value)
                # print("Mape sum: ", mape)
            mape = mape * 100 / len(Y_train)
            # print('epoch %s, MAPE %s, FITNESS %s' % (epoch, mape, fitness))
            # printNetworkWeight()
        return mape

    def predict(self, row):
        output = self.feedForward(row)
        return output

    def data_testing(self, X_test, Y_test, max_value, min_value):
        X_test = X_test.values.tolist()
        mape = 0
        for x, data_row in enumerate(Y_test):
            actual_value = un_normalized(Y_test.iloc[x], max_value, min_value)
            forecasted_value = un_normalized(
                self.predict(X_test[x])[0], max_value, min_value)
            # print('Test: ', actual_value, " ", forecasted_value)
            mape += abs((actual_value - forecasted_value) / actual_value)
            # print("Mape sum: ", mape)
        mape = mape * 100 / len(Y_test)
        # print('MAPE %s' % (mape))
        # printNetworkWeight()
        return mape
