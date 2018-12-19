from core.particle_swarm_optimization import Particle
from core.particle_swarm_optimization import ParticleSwarmOptimization
from core.backpropagation_neural_net import BackpropagationNN, Neuron
from random import random
import pandas as pd


dataset = pd.read_csv('dataset_minyak_kelapa_sawit.csv')
dataset_value = dataset.iloc[:, 1].values

normalized_dataset = list()
for x in range(len(dataset_value)):
    norm_value = (dataset_value[x] - min(dataset_value)) / (
        max(dataset_value) - min(dataset_value))
    normalized_dataset.append(norm_value)

time_series_dataset = [
    normalized_dataset[:len(normalized_dataset) - 5],
    normalized_dataset[1:len(normalized_dataset) - 4],
    normalized_dataset[2:len(normalized_dataset) - 3],
    normalized_dataset[3:len(normalized_dataset) - 2],
    normalized_dataset[4:len(normalized_dataset) - 1],
    normalized_dataset[5:]]

# show_time_series_dataset
print("Time Series")
print(pd.DataFrame([list(i) for i in zip(*time_series_dataset)]))

df = pd.DataFrame([list(i) for i in zip(*time_series_dataset)])

X_train = df.iloc[0:10, :5]
Y_train = df.iloc[0:10, 5]
X_test = df.iloc[71:80, :5]
Y_test = df.iloc[71:80, 5]

max_dataset = 1292.0
min_dataset = 538.0


class BackpropagationPSO(BackpropagationNN):
            """docstring for BackpropagationPSO"""

            def __init__(self, input, hidden, output, learning_rate):
                super().__init__(input, hidden, output, learning_rate)

            # particle representation:
            # [ v11, v21, v31, v41, v51,
            #   v12, v22, v32, v42, v52,
            #   v13, v23, v33, v43, v53 ]
            def initWeight(self, partikel):
                layer = list()
                partikel_dimens_idx = 0
                input_to_hidden = list()
                bias = 1

                for i in range(self.HIDDEN_LAYER):
                    w = list()
                    for j in range(self.INPUT_LAYER):
                        w.append(partikel[partikel_dimens_idx])
                        partikel_dimens_idx += 1
                    # bias terletak di index akhir
                    # bias bisa diakses dengan index w[-1]
                    w.append(bias)
                    input_to_hidden.append(Neuron(w))

                hidden_to_output = list()
                for i in range(self.OUTPUT_LAYER):
                    w = list()
                    for j in range(self.HIDDEN_LAYER):
                        w.append(random())
                    w.append(bias)
                    hidden_to_output.append(Neuron(w))

                layer.append(hidden_to_output)
                self.net = layer


class BackpropagationParticle(Particle):

    def __init__(self, particle_size):
        super().__init__(particle_size)

    def set_fitness(self):
        particle_position = self.position
        backPro = BackpropagationPSO(5, 3, 1, 0.4)
        backPro.initWeight(particle_position)
        backPro.training(X_train, Y_train, max_dataset, min_dataset, 2)
        mape = backPro.data_testing(
            X_test, Y_test, max_dataset, min_dataset)
        self.fitness = 100 / (100 + mape)
        # print("FITNESS: ", self.fitness)


class PSOxBackpro(ParticleSwarmOptimization):

    def __init__(self, pop_size, particle_size):
        super(PSOxBackpro, self).__init__(pop_size, particle_size)
        self.initPops(pop_size, particle_size)

    def initPops(self, pop_size, particle_size):
        self.pops = [BackpropagationParticle(
            particle_size) for n in range(pop_size)]
        self.p_best = self.pops
        self.g_best = self.get_g_best()


pso_backpp = PSOxBackpro(10, 15)
pso_backpp.optimize(10)
