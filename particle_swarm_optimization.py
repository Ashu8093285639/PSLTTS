from random import random

XMIN = 0
XMAX = 1


def velocity_boundary(k, xmin, xmax):
    vmax = k * (xmax - xmin) / 2
    vmin = -1 * vmax
    return (vmin, vmax)


def velocity_clamping(vnew, vmin, vmax):
    if(vnew > vmax):
        return vmax
    elif(vnew < vmin):
        return vmin
    else:
        return vnew


def position_clamping(xnew, xmin, xmax):
    if(xnew > xmax):
        return xmax
    elif(xnew < xmin):
        return xmin
    else:
        return xnew


VMIN, VMAX = velocity_boundary(0.6, XMIN, XMAX)


class Particle():

    XMAX = 1
    XMIN = 0

    def __init__(self, particle_size):

        self.position = [(XMIN + random() * (
            XMAX - XMIN)) for i in range(particle_size)]
        self.velocity = [0 for i in range(particle_size)]
        self.fitness = self.get_fitness(self.position)

    def get_fitness(particle_position):
        # net = list()
        # net = initWeightPSO(partikel)
        # fitness = training(net, X_train, Y_train, 10)
        fitness = 0
        for val in particle_position:
            fitness += val
        return fitness


class ParticleSwarmOptimization():

    def __init__(self, pop_size, particle_size):
        self.initPops(pop_size, particle_size)

    def initPops(self, pop_size, particle_size):
        self.pops = [Particle(particle_size) for n in range(pop_size)]
        self.p_best = self.pops
        self.g_best = self.get_g_best()

    def get_g_best(self):
        p_best_sorted = self.p_best
        p_best_sorted = sorted(
            p_best_sorted, key=lambda partc: partc.fitness, reverse=True)
        return p_best_sorted[0]

    def update_velocity_and_position(self, w, c1, c2):
        updated_pops = self.pops
        for partc_id, particle in enumerate(self.pops):
            for partc_id_dimen in range(len(particle.velocity)):
                #  update velocity
                vnew = (
                    w * particle.velocity[partc_id_dimen] +
                    c1 * random() *
                    (self.p_best[partc_id].position[partc_id_dimen] -
                        particle.position[partc_id_dimen]) +
                    c2 * random() +
                    (self.g_best.position[partc_id_dimen] -
                        particle.position[partc_id_dimen])
                )

                # update velocity
                vnew = velocity_clamping(vnew, VMIN, VMAX)
                updated_pops[partc_id].velocity[partc_id_dimen] = vnew

                # update position
                xnew = particle.position[partc_id_dimen] + vnew
                updated_pops[partc_id].position[partc_id_dimen] = xnew

            new_particle_fitness = self.get_fitness(
                updated_pops[partc_id].position)
            updated_pops[partc_id].fitness = new_particle_fitness

        self.pops = updated_pops

    def update_p_best(self):
        for partc_id in range(len(self.pops)):
            if(self.pops[partc_id].fitness > self.p_best[partc_id].fitness):
                self.p_best[partc_id] = self.pops[partc_id]
        # print(p_best)

    def optimize(self, tmax):
        t = 0
        while(t < tmax):
            w = 0.4 + (0.9 - 0.4) * ((tmax - t) / tmax)
            c1 = (0.5 - 2.5) * (t / tmax) + 2.5
            c2 = (0.5 - 2.5) * (t / tmax) + 0.5
            self.update_velocity_and_position(w, c1, c2)
            self.update_p_best()
            self.g_best = self.get_g_best()
            # print(p_best_pops)
            print(self.g_best.position)
            print(self.g_best.fitness)
            t += 1


pso = ParticleSwarmOptimization(10, 10)
pso.optimize(10)
# pops = initPopulasi(1, 22)
# p_best_pops = pops
# g_best = update_g_best(p_best_pops)
