from core.particle_swarm_optimization import Particle
from core.particle_swarm_optimization import ParticleSwarmOptimization
from core.backpropagation_neural_net import BackpropagationNN, Neuron
from random import random
# import pandas as pd
# from matplotlib import pyplot as plt


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

    def __init__(self, particle_size, x_train, y_train, x_test, y_test, max_val, min_val, epoch):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.max_val, self.min_val = max_val, min_val
        self.epoch = epoch
        super().__init__(particle_size)


    def calculate_fitness(self):

        particle_position = self.position

        backPro = BackpropagationPSO(5, 3, 1, 0.01)
        backPro.initWeight(particle_position)

        backPro.training(self.x_train, self.y_train, self.max_val, self.min_val, self.epoch)
        mape = backPro.data_testing(
            self.x_test, self.y_test, self.max_val, self.min_val)

        self.fitness = 100 / (100 + mape)


class PSOxBackpro(ParticleSwarmOptimization):

    def __init__(self, pop_size, particle_size, k=None):
        super(PSOxBackpro, self).__init__(pop_size, particle_size, k)

    def set_backpro_param(self, x_train, y_train, x_test, y_test, max_val, min_val, epoch):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.max_val, self.min_val = max_val, min_val
        self.epoch = epoch

    def initPops(self):
        self.pops = []
        self.pops = [BackpropagationParticle(
            self.particle_size, 
            self.x_train, 
            self.y_train, 
            self.x_test, 
            self.y_test, 
            self.max_val, 
            self.min_val, 
            self.epoch) for n in range(self.pop_size)]
        self.p_best = self.pops
        self.g_best = self.get_g_best()

