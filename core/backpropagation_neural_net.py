"""
Melakukan import fungsi abs, exp dan random
"""
from numpy import abs
from numpy import exp
from random import random


def un_normalized(value, min, max):
    """
    Fungsi unnormalized digunakan untuk mengembalikan nilai normalisasi
    ke nilai asli yang belum dinormalisasi
    """
    return value * (max - min) + min


class Neuron:
    """
        Object neuron merupakan objek yang menyimpan data berupa data bobot,
        nilai output dan data delta perubahan bobot
    """

    def __init__(self, weight):
        """
            Inisialisasi neuron yang terdiri dari list yang sebanyak
            jumlah bobot jaringan
        """
        self.weight = weight
        self.output = list()
        self.delta = list()


class BackpropagationNN:
    """
        Class utama dari algoritme backpropagation
    """

    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate):
        """
            Inisialisasi algortime backpro

            variabel input_layer, hidden_layer dan output_layer
            merupakan jumlah neuron yang diinputkan user untuk masing-masing
            layer
        """
        self.INPUT_LAYER = input_layer
        self.HIDDEN_LAYER = hidden_layer
        self.OUTPUT_LAYER = output_layer
        self.LEARNING_RATE = learning_rate

    def __str__(self):
        """
            hanya sebuah fungsi yang digunakan untuk menampilkan bobot jaringan
            pada semua neuron yang ada di objek backpropagation
        """
        str_buff = ""
        str_buff += "Input to Hidden\n"
        for neuron in self.net[0]:
            str_buff += str(neuron.weight) + "\n"
        str_buff += "Hidden to Output\n"
        for neuron in self.net[1]:
            str_buff += str(neuron.weight) + "\n"
        return str_buff

    def summingFunction(self, last_weight, training_data):
        """
            Summing function pada backpropagation
            input berupa bobot pada jaringan backpro
            dan bobot baris training data
        """
        bias = last_weight[-1]  # bias on last index
        y_in = 0
        for idx in range(0, len(last_weight) - 1):
            y_in += last_weight[idx] * training_data[idx]
        return bias + y_in

    def activation(self, y_in):
        """
            Hasil dari summing function akan dilakukan proses aktivasi
        """
        return 1.0 / (1.0 + exp(-1 * y_in))

    def derivativeFunction(self, y):
        """
            Fungsi ini merupakan fungsi yang dijalankan ketika backpropagate
        """
        return y * (1.0 - y)

    def initWeight(self):
        """
            Melakukan inisialisasi bobot
            Secara default, bobot diinisialiasi secara random dan jumlah neuron
            sesuai dengan jumlah input, hidden dan output yang ditentukan
            oleh user
        """

        layer = list()
        input_to_hidden = [
            Neuron([random() for i in range(
                self.INPUT_LAYER + 1)]) for j in range(self.HIDDEN_LAYER)]
        layer.append(input_to_hidden)
        hidden_to_output = [
            Neuron([random() for i in range(
                self.HIDDEN_LAYER + 1)]) for j in range(self.OUTPUT_LAYER)]
        layer.append(hidden_to_output)
        """
            Bobot yang diinisialisasi akan disimpan pada variabel net
            Sehingga, variabel net merupakan list dengan struktur
            self.net = [input, hidden, output]
        """
        self.net = layer

    def feedForward(self, row):
        """ Melakukan proses feedForward dengan data inputan berupa
            data baris timeseries
        """
        processed_layer = row
        # feedforward dilakukan dengan memproses setiap layer yang ada di
        # variabel lokal self.net
        for layer in self.net:
            # print(layer)
            new_inputs = list()
            for neuron in layer:

                # melakukan proses summing function dan hasilnya akan dilakukan
                # aktivasi
                y_in = self.summingFunction(neuron.weight, processed_layer)
                neuron.output = self.activation(y_in)
                new_inputs.append(neuron.output)
            processed_layer = new_inputs
        return processed_layer  # nilai y1, y2, .... yx

    def backPpError(self, target):
        """
            Proses backpropagated dilakukan secara mundur dari output hingga
            ke input layer (reversed net)
        """
        for id_layer in reversed(range(len(self.net))):
            layer = self.net[id_layer]
            # print(id_layer)
            error_factor = list()
            # error factor adalah selisih antara target terhadap output
            if id_layer == len(self.net) - 1:
                """
                    kondisi yang dijalankan jika layer yang dipilih
                    yaitu layer output.
                    error factor didapatkan dari target - output
                """
                for weight_id in range(len(layer)):
                    neuron = layer[weight_id]
                    error_factor.append(target[weight_id] - neuron.output)
            else:
                """
                    kondisi yang dijalankan jika layer yang dipilih
                    yaitu layer selain output.
                    error factor didapatkan dari hasil sumproduct antara bobot
                    layer sebelum dengan bobot layer yang aktif
                """
                for weight_id in range(len(layer)):
                    error_factor_sum = 0.0
                    # prev_layer menunjukkan bahwa yang diproses adalah
                    # layer sebelumnya
                    prev_layer = self.net[id_layer - 1]
                    for neuron in prev_layer:
                        # neuron.delta merupakan nilai delta hasil dari
                        # derivative function yang ada di bawah
                        # (fungsi for)
                        error_factor_sum += neuron.weight[weight_id] * neuron.delta
                    error_factor.append(error_factor_sum)

            for weight_id in range(len(layer)):
                neuron = layer[weight_id]
                neuron.delta = error_factor[weight_id] * self.derivativeFunction(neuron.output)

    def updateWeight(self, row):
        for i in range(len(self.net)):
            inputs = list()
            if i == 0:
                """
                    Kondisi ketika layer yang diproses adalah
                    input to hidden
                """
                inputs = row
            else:
                """
                    Kondisi ketika layer yang diproses adalah
                    hidden to output
                """
                prev_net = self.net[i - 1]
                inputs = [neuron.output for neuron in prev_net]

            """
                setelah fungsi backpropagate dijalankan. Nilai
                delta akan disimpan pada masing-masing neuron
                dan siap diakses untuk melakukan proses update bobot
            """
            for neuron in self.net[i]:
                neuron.weight[-1] += self.LEARNING_RATE * neuron.delta
                for weight_id in range(len(neuron.weight) - 1):
                    neuron.weight[weight_id] += self.LEARNING_RATE * (
                        neuron.delta - inputs[weight_id])

    def training(self, X_train, Y_train, max_value, min_value, total_epoch):
        """
            nilai max_value dan min_value digunakan untuk
            perhitungan nilai mape (karena perhitungan nilai mape
            harus menggunakan data yang tidak dinormalisasi)

            Proses training menggunakan data input (X_train)
            dan output (Y_train) sebanyak total_epoch
        """

        X_list = X_train.values.tolist()
        Y_list = [[x] for x in Y_train.values.tolist()]

        # perulangan sebanyak jumlah epoch
        for epoch in range(total_epoch):
            mape = 0
            for idx, data_row in enumerate(Y_list):
                # perulangan sebanyak jumlah data
                self.feedForward(X_list[idx])
                self.backPpError(Y_list[idx])
                self.updateWeight(X_list[idx])

            for x in range(len(Y_list)):
                # penghitungan nilai mape
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
        """
            proses predict hanya menjalankan fungsi feedforward dan
            mendapatkan output dari feedforward tersebut
        """
        output = self.feedForward(row)
        return output

    def data_testing(self, X_test, Y_test, max_value, min_value):
        """
            nilai max_value dan min_value digunakan untuk
            perhitungan nilai mape (karena perhitungan nilai mape
            harus menggunakan data yang tidak dinormalisasi)

            Proses testing menggunakan data input (X_test)
            dan output (Y_test). Hasil testing berupa nilai mape
        """
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


# Contoh penggunaan class Backpropagation
backpp = BackpropagationNN(5, 5, 1, 0.5)
backpp.initWeight()
backpp.training(
    data_training, output_data_training, nilai_maksimal_dataset, nilai_minimal_dataset, epoch)
nilai_mape = backpp.data_testing(
    data_testing, output_data_testing, nilai_maksimal_dataset, nilai_minimal_dataset)
fitness = 100 / (100 + nilai_mape)
