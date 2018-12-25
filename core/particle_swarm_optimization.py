# melakukan import class random
from random import random


class Particle():
    """Particle pada object PSO direpresentasikan sebagai object yang memiliki
    nilai posisi, velocity dan nilai fitness
    """

    def __init__(self, particle_size):
        """ melakukan inisialisasi posisi particle secara random, nilai
        velocity awal diset ke 0. proses init juga sekaligus menetukan nilai
        fitness pada partikel awal yang telah diinisialisasi
        """
        self.position = [random() for i in range(particle_size)]
        self.velocity = [0 for i in range(particle_size)]
        self.set_fitness()

    def set_fitness(self):
        """ secara default, nilai fitness akan diset ke 0. namun pada
        implementasinya, fungsi ini dapat dioverride atau dikustomisasi
        menjadi fungsi lain (lihat file main.py) agar lebih fleksibel
        """
        self.fitness = 0


class ParticleSwarmOptimization():

    def __init__(self, pop_size, particle_size, k=None):
        """ Melakukan inisialisasi object PSO dengan parameter berapa jumlah populasi
        (pop_size), ukuran dimensi partikel (particle_size) dan nilai
        k-velocity clamping. secara default nilai k-velocity
        clamping diset ke None (default tidak menggunakan velocity clamping)

        nilai k akan disimpan pada variabel object PSO

        ket: penggunaan self menunjukkan bahwa masing-masing
        atribut akan disimpan pada object (sebagai variabel dalam class)
        fungsinya hampir mirip 'this' pada java
        """

        self.initPops(pop_size, particle_size)
        self.k = k

    def initPops(self, pop_size, particle_size):
        """ Fungsi yang digunakan untuk membuat partikel awal dari pso.
        """

        # membuat list object particle sebanyak jumlah popsize yang
        # dimasukkan user
        self.pops = [Particle(particle_size) for n in range(pop_size)]

        # pbest awal yaitu populasi yang telah diinisialisasi
        self.p_best = self.pops

        # gbest didapatkan dengan memanggil fungsi get_gbest
        # yang ada di class PSO dan disimpan di local variabel g_best
        self.g_best = self.get_g_best()

    def get_g_best(self):
        """
        Mengambil object particle yang memiliki nilai fitness paling tinggi
        """

        p_best_sorted = self.p_best
        # prosesnya dilakukan dengan melakukan sorting descending berdasarkan
        # nilai fitness pada object particle (x.fitness)
        p_best_sorted.sort(key=lambda x: x.fitness, reverse=True)
        # nilai particle paling tinggi (indeks ke 0) yang akan dikembalikan
        return p_best_sorted[0]

    def velocity_clamping(self, vnew):
        """Velocity clamping merupakan pembatasan nilai velocity
        (lihat di PSO real-coded)
        """

        # jika nilai k tidak diset, maka tidak dilakukan velocity clamping
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
        """
        Fungsi yang dilakukan untuk melakukan update posisi dan kecepatan untuk
        semua partikel yang ada di object class PSO
        """

        updated_pops = self.pops
        for partc_id, particle in enumerate(self.pops):
            """
                Perulangan untuk:
                pops:
                 - partikel 1
                 - partikel 2
                 - partikel ke n
            """
            for partc_id_dimen in range(len(particle.velocity)):
                """
                    perulangan untuk:
                    partikel 1:
                        - ['dimensi1', dimensi2, dimensi3
                            dimensi-n]
                    ket: n sesuai dengan panjang partikel yang diinputkan user
                """

                """ update velocity
                    masing-masing dimensi akan dilakukan proses penghitungan
                    vnew atau velocity baru (lihat persamaan update kecepatan
                    pada real-coded PSO)
                """
                vnew = (
                    w * particle.velocity[partc_id_dimen] +
                    c1 * random() *
                    (self.p_best[partc_id].position[partc_id_dimen] -
                        particle.position[partc_id_dimen]) +
                    c2 * random() +
                    (self.g_best.position[partc_id_dimen] -
                        particle.position[partc_id_dimen])
                )

                # melakukan pembatasan kecepatan atau velocity clamping
                if self.k is not None:
                    vnew = self.velocity_clamping(vnew)

                # menyimpan nilai kecepatan tiap-tiap partikel pada atribut
                # velocity (sesuai yang ada di class Particle)
                updated_pops[partc_id].velocity[partc_id_dimen] = vnew

                # update position
                # dilakukan dengan menambahkan velocity baru ke posisi
                xnew = particle.position[partc_id_dimen] + vnew
                updated_pops[partc_id].position[partc_id_dimen] = xnew

            # Setelah nilai posisi diupdate, maka yang harus dilakukan yaitu
            # menentukan nilai fitness untuk masing-masing partikel
            updated_pops[partc_id].set_fitness()

        # mengganti partikel yang lama dengan partikel yang telah diupdate
        self.pops = updated_pops

    def update_p_best(self):
        """ fungsi ini digunakan untuk mengupdate variabel p_best yang ada di class
        PSO. Prosesnya yaitu dilakukan dengan membandingkan nilai p_best dan
        populasi yang telah diupdate. Jika nilai fitness pada salah satu
        partikel, lebih besar dibanding nilai pada p_bestnya.
        maka posisi partikel saat itu akan menjadi p_best yang baru
        """

        for partc_id in range(len(self.pops)):
            if(self.pops[partc_id].fitness > self.p_best[partc_id].fitness):
                self.p_best[partc_id] = self.pops[partc_id]

    def get_average_fitness(self):
        """
            mendapatkan rata-rata fitness dari keseluruhan partikel pada
            populasi
        """
        sum_fitness = 0
        for partc in self.pops:
            sum_fitness += partc.fitness
        return sum_fitness / len(self.pops)

    def optimize(self, tmax, w, c1, c2):
        """ keseluruhan fungsi PSO dijalankan pada method ini
        """
        t = 0   # t awal diset ke 0
        print("Popsize: ", len(self.pops), ", Itermax: ", tmax)
        print("w: ", w, ", c1: ", c1, ", c2: ", c2)
        print("k: ", self.k)

        # mencetak nilai fitness semua partikel pada populasi
        for p in self.pops:
            print(p.fitness)

        # perulangan sebanyak nilai tmax
        while(t < tmax):
            # w = 0.4 + (0.9 - 0.4) * ((tmax - t) / tmax)
            # c1 = (0.5 - 2.5) * (t / tmax) + 2.5
            # c2 = (0.5 - 2.5) * (t / tmax) + 0.5

            # melakukan proses update velocity dan posisi
            self.update_velocity_and_position(w, c1, c2)

            # menentukan p_best dengan menjalankan fungsi update p_best
            self.update_p_best()

            # menentukan g_best dengan menjalankan get_gbest
            self.g_best = self.get_g_best()
            # print(p_best_pops)
            # print(self.g_best.position)
            print("-------------------------------------------------------")
            print("t = ", (t + 1))
            print("Global Best Position: ", self.g_best.position)
            print("Global Best Particle Velocity: ", self.g_best.velocity)
            print("Fitness: ", self.g_best.fitness)
            print("Average Fitness: ", self.get_average_fitness())
            print("-------------------------------------------------------")
            t += 1
        # setelah training selesai, maka akan mengambilak nilai fitness g_best
        # dan fitness rata-rata
        return self.g_best.fitness, self.get_average_fitness()


# Contoh penggunaan class particle swarm optimization

# inisialisasi object pso
pso = ParticleSwarmOptimization(10, 10, 1)

# menjalankan fungsi optimize atau training
pso.optimize(10, 1, 5, 1)
