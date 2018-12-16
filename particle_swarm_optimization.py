import random as rd


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


def get_fitness(partikel):
    net = list()
    net = initWeightPSO(partikel)
    fitness = training(net, X_train, Y_train, 10)
    return fitness


def initPopulasi(pop_size, partikel_size):
    pop = [{
        "position":
        [(XMIN + rd.random() * (XMAX - XMIN)) for i in range(partikel_size)],
        "velocity":
        [0 for i in range(partikel_size)]} for n in range(pop_size)]
    for indv_idx in range(len(pop)):
        pop[indv_idx]["fitness"] = get_fitness(pop[indv_idx]["position"])
    return pop


def update_velocity_and_position(pops, p_best_pops, g_best, w, c1, c2):
    new_pops = list()

    for id_partikel, partikel in enumerate(pops):
        new_particle_velocity = list()
        new_particle_position = list()
        for id_dimensi in range(len(partikel['velocity'])):
            #  update velocity
            vnew = (
                w * partikel['velocity'][id_dimensi] +
                c1 * rd.random() *
                (p_best_pops[id_partikel]['position'][id_dimensi] -
                    partikel['position'][id_dimensi]) +
                c2 * rd.random() +
                (g_best['position'][id_dimensi] -
                    partikel['position'][id_dimensi])
            )
            vnew = velocity_clamping(vnew, VMIN, VMAX)
            # print(vnew)
            new_particle_velocity.append(vnew)

            #  update position
            xnew = partikel['position'][id_dimensi] + vnew
            new_particle_position.append(xnew)

        new_particle_fitness = get_fitness(new_particle_position)
        new_pops.append({
            "position": new_particle_position,
            "velocity": new_particle_velocity,
            "fitness": new_particle_fitness
        })

    return new_pops


def update_p_best(pops, new_pops):
    p_best = pops
    for idx in range(len(pops)):
        if(new_pops[idx]['fitness'] > pops[idx]['fitness']):
            p_best[idx] = new_pops[idx]
    # print(p_best)
    return p_best


def update_g_best(pbest_pops):
    sorted_pbest = sorted(pbest_pops, key=lambda k: k['fitness'], reverse=True)
    return sorted_pbest[0]


pops = initPopulasi(1, 22)
p_best_pops = pops
g_best = update_g_best(p_best_pops)

t = 0
tmax = 10
while(t < tmax):
    w = 0.4 + (0.9 - 0.4) * ((tmax - t) / tmax)
    c1 = (0.5 - 2.5) * (t / tmax) + 2.5
    c2 = (0.5 - 2.5) * (t / tmax) + 0.5
    new_pops = update_velocity_and_position(
        pops, p_best_pops, g_best, w, c1, c2)
    p_best_pops = update_p_best(p_best_pops, new_pops)
    # print(p_best_pops)
    pops = p_best_pops
    g_best = update_g_best(pops)
    # print(g_best)
    print(pops)
    t += 1
