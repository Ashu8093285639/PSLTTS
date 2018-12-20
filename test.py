from core.particle_swarm_optimization import Particle
from core.particle_swarm_optimization import ParticleSwarmOptimization
from core.backpropagation_neural_net import BackpropagationNN, Neuron
from random import random
import pandas as pd


dataset_value = [562, 572, 598, 702, 801, 726, 639, 723, 674, 680]

max_dataset = max(dataset_value)
min_dataset = min(dataset_value)

normalized_dataset = list()
for x in range(len(dataset_value)):
    norm_value = (dataset_value[x] - min(dataset_value)) / (
        max(dataset_value) - min(dataset_value))
    normalized_dataset.append(norm_value)

time_series_dataset = [
    dataset_value[:len(dataset_value) - 4],
    dataset_value[1:len(dataset_value) - 3],
    dataset_value[2:len(dataset_value) - 2],
    dataset_value[3:len(dataset_value) - 1],
    dataset_value[4:len(dataset_value)]]

time_series_dataset_normalized = [
    normalized_dataset[:len(normalized_dataset) - 4],
    normalized_dataset[1:len(normalized_dataset) - 3],
    normalized_dataset[2:len(normalized_dataset) - 2],
    normalized_dataset[3:len(normalized_dataset) - 1],
    normalized_dataset[4:len(normalized_dataset)]]

df = pd.DataFrame(time_series_dataset_normalized)

X_train = df.iloc[0:3, :5]
Y_train = df.iloc[0:3, 5]
X_test = df.iloc[3:5, :5]
Y_test = df.iloc[3:5, 5]


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
                        w.append(0.1)
                    w.append(bias)
                    hidden_to_output.append(Neuron(w))

                layer.append(hidden_to_output)
                self.net = layer


class BackpropagationParticle(Particle):

    def __init__(self, particle_size, custom_position):
        super().__init__(particle_size)

    def set_fitness(self):
        particle_position = self.position
        backPro = BackpropagationPSO(5, 1, 1, 0.5)
        backPro.initWeight(particle_position)
        backPro.training(X_train, Y_train, max_dataset, min_dataset, 4)
        mape = backPro.data_testing(
            X_train, Y_train, max_dataset, min_dataset)
        self.fitness = 100 / (100 + mape)
        print("MAPE: ", mape)


class PSOxBackpro(ParticleSwarmOptimization):

    def __init__(self, pop_size, particle_size):
        super(PSOxBackpro, self).__init__(pop_size, particle_size)
        self.initPops(pop_size, particle_size)

    def initPops(self, pop_size, particle_size):
        self.pops = [
            BackpropagationParticle(5,
                [0.577641713944198, 0.710593877978217, 0.268382608759389, 0.779104550645483, 0.053359199909105]),
            BackpropagationParticle(5,
                [0.175734154487125, 0.041676622340747, 0.917352950461798, 0.542852477162069, 0.744185882314755]),
            BackpropagationParticle(5,
                [0.342475594273213, 0.980766647404581, 0.346280857556764, 0.425169813861027, 0.138667316313456]),
            BackpropagationParticle(5,
                [0.376212632068454, 0.915369338264335, 0.907338954166647, 0.411223670835483, 0.805821893303824]),
            BackpropagationParticle(5,
                [0.498188770596265, 0.147908292312007, 0.561066290268070, 0.529655809230534, 0.170343903942480])
        ]
        self.p_best = self.pops
        self.g_best = self.get_g_best()

    def optimize(self, tmax):
        t = 0
        print()
        print("-------------------------------------------------------")
        print("Initialization Phase")
        print("-------------------------------------------------------")
        print("t = ", (t + 0))
        print("Global Best Position: ", self.g_best.position)
        print("Global Best Particle Velocity: ", self.g_best.velocity)
        print("Fitness: ", self.g_best.fitness)
        print("-------------------------------------------------------")
        while(t < tmax):
            w = 0.4 + (0.9 - 0.4) * ((tmax - t) / tmax)
            c1 = (0.5 - 2.5) * (t / tmax) + 2.5
            c2 = (0.5 - 2.5) * (t / tmax) + 0.5
            self.update_velocity_and_position(w, c1, c2)
            self.update_p_best()
            self.g_best = self.get_g_best()
            # print(p_best_pops)
            # print(self.g_best.position)
            print("-------------------------------------------------------")
            print("t = ", (t + 1))
            print("Global Best Position: ", self.g_best.position)
            print("Global Best Particle Velocity: ", self.g_best.velocity)
            print("Fitness: ", self.g_best.fitness)
            print("-------------------------------------------------------")
            t += 1


pso_backpp = PSOxBackpro(5, 5)
pso_backpp.optimize(10)


t = BackpropagationParticle(5,
                [0.577641713944198, 0.710593877978217, 0.268382608759389, 0.779104550645483, 0.053359199909105])
print(t.fitness)