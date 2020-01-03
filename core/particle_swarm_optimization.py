from random import random


class Particle():

    def __init__(self, particle_size):
        self.position = [random() for i in range(particle_size)]
        self.velocity = [0 for i in range(particle_size)]
        self.calculate_fitness()

    def calculate_fitness(self):
        self.fitness = 0


class ParticleSwarmOptimization():

    def __init__(self, pop_size, particle_size, k=None):
        self.k = k
        self.pop_size, self.particle_size = pop_size, particle_size

    def initPops(self):
        self.pops = [Particle(self.particle_size) for n in range(self.pop_size)]
        self.p_best = self.pops
        self.g_best = self.get_g_best()

    def get_g_best(self):

        p_best_sorted = self.p_best
        p_best_sorted.sort(key=lambda x: x.fitness, reverse=True)
        return p_best_sorted[0]

    def velocity_clamping(self, vnew):
        if self.k is None:
            return vnew
        else:
            vmax = self.k * (1 - 0) / 2
            vmin = -1 * vmax
            if(vnew > vmax):
                return vmax
            elif(vnew < vmin):
                return vmin
            else:
                return vnew

    def update_velocity_and_position(self, w, c1, c2):
        updated_pops = self.pops
        for partc_id, particle in enumerate(self.pops):
            for partc_id_dimen in range(len(particle.velocity)):
                vnew = (
                    w * particle.velocity[partc_id_dimen] +
                    c1 * random() *
                    (self.p_best[partc_id].position[partc_id_dimen] -
                        particle.position[partc_id_dimen]) +
                    c2 * random() +
                    (self.g_best.position[partc_id_dimen] -
                        particle.position[partc_id_dimen])
                )

                if self.k is not None:
                    vnew = self.velocity_clamping(vnew)

                updated_pops[partc_id].velocity[partc_id_dimen] = vnew

                xnew = particle.position[partc_id_dimen] + vnew
                updated_pops[partc_id].position[partc_id_dimen] = xnew

            updated_pops[partc_id].calculate_fitness()

        # mengganti partikel yang lama dengan partikel yang telah diupdate
        self.pops = updated_pops

    def update_p_best(self):
        for partc_id in range(len(self.pops)):
            if(self.pops[partc_id].fitness > self.p_best[partc_id].fitness):
                self.p_best[partc_id] = self.pops[partc_id]

    def get_average_fitness(self):
        sum_fitness = 0
        for partc in self.pops:
            sum_fitness += partc.fitness
        return sum_fitness / len(self.pops)

    def optimize(self, tmax, w, c1, c2):
        t = 0   # t awal diset ke 0
        # print("Popsize: ", len(self.pops), ", Itermax: ", tmax)
        # print("w: ", w, ", c1: ", c1, ", c2: ", c2)
        # print("k: ", self.k)

        # for p in self.pops:
        #     print(p.fitness)

        while(t < tmax):
            self.update_velocity_and_position(w, c1, c2)
            self.update_p_best()
            self.g_best = self.get_g_best()
            # print("-------------------------------------------------------")
            # print("t = ", (t + 1))
            # print("Global Best Position: ", self.g_best.position)
            # print("Global Best Particle Velocity: ", self.g_best.velocity)
            # print("Global Best Fitness: ", self.g_best.fitness)
            # print("Average Fitness: ", self.get_average_fitness())
            # print("-------------------------------------------------------")
            t += 1
        return self.g_best.fitness, self.get_average_fitness()
