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

print("Time Series")
print(pd.DataFrame([list(i) for i in zip(*time_series_dataset)]))

df = pd.DataFrame([list(i) for i in zip(*time_series_dataset)])


class BackpropagationPSO(BackpropagationNN):

    def __init__(self, input, hidden, output, learning_rate):
        super().__init__(input, hidden, output, learning_rate)

    def initWeight(self, partikel):
        layer = list()
        partikel_dimens_idx = 0
        input_to_hidden = list()

        for i in range(self.HIDDEN_LAYER):
            w = list()
            for j in range(self.INPUT_LAYER):
                w.append(partikel[partikel_dimens_idx])
                partikel_dimens_idx += 1

            w.append(random())
            input_to_hidden.append(Neuron(w))
        layer.append(input_to_hidden)

        hidden_to_output = list()
        for i in range(self.OUTPUT_LAYER):
            w = list()
            for j in range(self.HIDDEN_LAYER + 1):
                w.append(random())
            hidden_to_output.append(Neuron(w))

        layer.append(hidden_to_output)
        self.net = layer


class BackpropagationParticle(Particle):

    def __init__(self, particle_size):
        super().__init__(particle_size)

    def set_fitness(self):

        particle_position = self.position

        backPro = BackpropagationPSO(5, 3, 1, 0.01)
        backPro.initWeight(particle_position)

        backPro.training(X_train, Y_train, max_dataset, min_dataset, 40)
        mape = backPro.data_testing(
            X_test, Y_test, max_dataset, min_dataset)

        self.fitness = 100 / (100 + mape)


class PSOxBackpro(ParticleSwarmOptimization):

    def __init__(self, pop_size, particle_size, k=None):
        super(PSOxBackpro, self).__init__(pop_size, particle_size, k)
        self.initPops(pop_size, particle_size)

    def initPops(self, pop_size, particle_size):
        self.pops = [BackpropagationParticle(
            particle_size) for n in range(pop_size)]
        self.p_best = self.pops
        self.g_best = self.get_g_best()


X_train = df.iloc[:, :5]
Y_train = df.iloc[:, 5]
X_test = df.iloc[:, :5]
Y_test = df.iloc[:, 5]

pso_backpp = PSOxBackpro(10, 15, 1)  # populasi 10, dimensi partikel 15, k = 1
gbest_fitness, avg_fitness = pso_backpp.optimize(3, 1, 1, 1)

max_dataset = 1292.0
min_dataset = 538.0


fitness = ""
epoch_parameter_test = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]

for epoch in range(len(epoch_parameter_test)):
    fitness += str(epoch_parameter_test[epoch]) + ", "
    for i in range(10):
        backpro = BackpropagationNN(5, 3, 1, 0.01)
        backpro.initWeight()
        backpro.training(
            X_train, Y_train,
            max_dataset, min_dataset, epoch_parameter_test[epoch])
        mape = backpro.data_testing(X_train, Y_train, max_dataset, min_dataset)
        fitness += str((100 / (100 + mape))) + ", "
    fitness += "\n"
print()
print(fitness)


# Contoh Sample Pengujian PSO

# menyimpan hasil pengujian dalam format string
g_best_fitness = ""
average_fitness = ""

# Pengujian jumlah iterasi PSO
iter_param_test = [5, 10, 15, 20, 25, 30, 35, 40, 45]
for iterasi in range(len(iter_param_test)):
    g_best_fitness += str(iter_param_test[iterasi]) + ", "
    average_fitness += str(iter_param_test[iterasi]) + ", "
    for i in range(10):
        # pengujian dilakukan 10 kali
        pso_backpp = PSOxBackpro(10, 15, 1)
        gbest_fitness, avg_fitness = pso_backpp.optimize(
            iter_param_test[iterasi], 1, 1, 1)
        g_best_fitness += str(gbest_fitness) + ", "
        average_fitness += str(avg_fitness) + ", "
    g_best_fitness += "\n"
    average_fitness += "\n"

# Pengujian jumlah partikel PSO
part_size_param_test = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
for part_size in range(len(part_size_param_test)):
    g_best_fitness += str(part_size_param_test[part_size]) + ", "
    average_fitness += str(part_size_param_test[part_size]) + ", "
    for i in range(10):
        pso_backpp = PSOxBackpro(part_size_param_test[part_size], 15, 1)
        gbest_fitness, avg_fitness = pso_backpp.optimize(25, 1, 1, 1)
        g_best_fitness += str(gbest_fitness) + ", "
        average_fitness += str(avg_fitness) + ", "
    g_best_fitness += "\n"
    average_fitness += "\n"

# Pengujian kombinasi nilai c1 dan c2
c1 = [2.5, 2, 1.5, 1, 0.5]
c2 = [0.5, 1, 1.5, 2, 2.5]
for c_id in range(len(c1)):
    g_best_fitness += str(c1[c_id]) + ":" + str(c2[c_id]) + ", "
    average_fitness += str(c1[c_id]) + ":" + str(c2[c_id]) + ", "
    for i in range(10):
        pso_backpp = PSOxBackpro(40, 15, 1)
        gbest_fitness, avg_fitness = pso_backpp.optimize(
            25, 1, c1[c_id], c2[c_id])
        g_best_fitness += str(gbest_fitness) + ", "
        average_fitness += str(avg_fitness) + ", "
    g_best_fitness += "\n"
    average_fitness += "\n"

# Pengujian nilai w
w = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for w_id in range(len(w)):
    g_best_fitness += str(w[w_id]) + ", "
    average_fitness += str(w[w_id]) + ", "
    for i in range(10):
        pso_backpp = PSOxBackpro(40, 15, 1)
        gbest_fitness, avg_fitness = pso_backpp.optimize(
            25, w[w_id], 2.5, 0.5)
        g_best_fitness += str(gbest_fitness) + ", "
        average_fitness += str(avg_fitness) + ", "
    g_best_fitness += "\n"
    average_fitness += "\n"


# Pengujian nilai k-velocity clamping
k = [0.2, 0.4, 0.6, 0.8, 1]
for k_id in range(len(k)):
    g_best_fitness += str(k[k_id]) + ", "
    average_fitness += str(k[k_id]) + ", "
    for i in range(10):
        pso_backpp = PSOxBackpro(40, 15, k[k_id])
        gbest_fitness, avg_fitness = pso_backpp.optimize(
            25, 0.4, 2.5, 0.5)
        g_best_fitness += str(gbest_fitness) + ", "
        average_fitness += str(avg_fitness) + ", "
    g_best_fitness += "\n"
    average_fitness += "\n"

# cetak hasil pengujian
print("GBest Fitness")
print(g_best_fitness)
print("Average Fitness")
print(average_fitness)
