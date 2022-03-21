import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndi

#  Load seismic trace
#  define parameters: number of spikes, number of optimisation iterations,
#  add first rc
#  calculate objective function
from thin_bed import *
#  from SYnthetic_case import imp, synthetic, wavelet
# model
tmax = 10 * 1e-3  # 10ms
tmin = 0
# impedance range
max_guess, min_guess = 2500, 1300
# wavelet parameters
f = 50
length = 0.1
dt = 1e-5
# Input seismogram
lenSeis = 999 + 1
# Optimisation criterion
L1 = True
# genetic algorithm
popSize = 400
xoverRate = 0.7
mutationRate = 0.3
b = 3   # 越大越不均匀
Generations = 150
##############################
Nsamp = int((tmax - tmin) / dt) + 1


def initiatePop(pop_size, nsamp):
    imp_pop = []
    for i in range(0, pop_size):
        indi = []
        for j in range(0, nsamp):
            indi.append(np.random.uniform(1200, 2500))
            imp_smooth = ndi.uniform_filter1d(indi, size=3)
        imp_pop.append(imp_smooth)
    imp_pop = np.asarray(imp_pop)
    return imp_pop


def calcuRc(imp_pop):
    rc_pop = []
    nmodel = np.shape(imp_pop)[0]
    nsamp = np.shape(imp_pop)[1]
    for i in range(nmodel):
        rc = []
        imp_individual = imp_pop[i]
        for j in range(nsamp - 1):
            rc.append((imp_individual[j] - imp_individual[j - 1]) / (imp_individual[j] + imp_individual[j - 1]))
        rc_pop.append(rc)
    return rc_pop


def generateSynthetic(rc_pop, wvlt):
    synthetic_pop = []
    for i in range(np.shape(rc_pop)[0]):
        trace = np.convolve(rc_pop[i], wvlt, mode='same')
        trace_norm = trace / max(trace)
        synthetic_pop.append(trace_norm)
    return synthetic_pop  # normalised synthetic trace population


def rankFitness(synthetic_norm, trace_norm):
    fitness = {}
    error = []
    synthetic_norm = np.asarray(synthetic_norm)
    trace_norm = np.asarray(trace_norm)
    for i in range(synthetic_norm.shape[0]):
        diff = synthetic_norm[i] - trace_norm
        error.append(sum(abs(diff)))  # 该条trace的总residual
    for j in range(len(error)):
        fitness[j] = sum(error) / error[j]
    fit = fitness.values()
    ave_fitness = sum(fit) / popSize
    ''' 
        err = []
        diff = synthetic_norm[i] - trace_norm
        for j in range(len(diff)):
            err.append(abs(diff[j] / trace_norm[j]))   # 该条trace上各点的percentage error
        print(err)
        error.append(sum(err) / len(err))    # 每条trace的percentage error，计算方式：average percentage error per sample
        # error[i] = sum(np.square(diff))  # L2-norm
    for k in range(len(error)):
        fitness[k] = sum(error) / error[k]
    fit = fitness.values()
    ave_fitness = sum(fit) / popSize
    '''
    return ave_fitness, error, sorted(fitness.items(), key=lambda item: item[1], reverse=True)


def selection(ave, rank_result):
    selection_results = []
    elite_size = 0
    for i in range(len(rank_result)):
        if rank_result[i][1] > ave:           # 控制elite_size，可以调整以在增加或减少crossover强度
            elite_size = elite_size + 1
            selection_results.append(rank_result[i][0])  # better than average are carried forward
    df = pd.DataFrame(np.array(rank_result), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_per'] = 100 * df.cum_sum / df.Fitness.sum()
    for i in range(popSize - elite_size):  # 需要补足的个体数
        pick = 100 * random.random()
        for k in range(len(rank_result)):
            if pick <= df.iat[k, 3]:  # 从前往后依次挑选
                selection_results.append(rank_result[k][0])
                break
    return elite_size, selection_results


def matingPool(population, selection_results):
    matingpool = []
    for i in range(len(selection_results)):
        index = selection_results[i]
        matingpool.append(population[index])
    return matingpool


def breedPopulation(matingpool, elite_size, xover_rate):
    children = []
    mating_pool = np.asarray(matingpool)
    pool = random.sample(matingpool, len(matingpool))  # 打乱impedance model pool
    for i in range(elite_size):
        children.append(mating_pool[i])
    for j in range(len(matingpool) - elite_size):
        if random.random() < xover_rate:
            rd = random.random()
            child = rd * pool[j] + (1 - rd) * pool[len(matingpool) - j - 1]
            children.append(child)
        else:
            children.append(pool[j])
    return children


def mutatePopulation(population, generation):
    mutated_pop = []
    population = np.asarray(population)
    for i in range(len(population)):
        indiv = population[i]
        if random.random() > mutationRate:  # populated into next generation without mutation
            mutated_pop.append(indiv)
            continue
        else:
            for j in range(len(indiv)):  # mutate according to mutateRate
                cut = 1 - (generation / Generations) ** b
                if random.random() < 0.5:  # tau = 0
                    indiv[j] += (max_guess - indiv[j]) * (1 - random.random() ** cut)
                else:  # tau =1
                    indiv[j] -= (indiv[j] - min_guess) * (1 - random.random() ** cut)
            mutated_pop.append(indiv)
    return mutated_pop


def nextGeneration(current_gen, generation, field_seis):
    current_rc = calcuRc(current_gen)
    current_seis = generateSynthetic(current_rc, wavelet)
    ave_fitness, error, pop_ranked = rankFitness(current_seis, field_seis)

    best_indi = current_gen[pop_ranked[0][0]]
    imp_residual = best_indi - imp
    err = []
    for j in range(len(imp_residual)):
        err.append(abs(imp_residual[j] / imp[j]))
    residual = sum(err) / len(err)  # 每个model的error，计算方式：average percentage error per sample 1/2
    # residual = sum(abs(imp_residual)) / sum(imp)  # 两种误差计算方式2/2
    print('trace error: ' + str(min(error)))  # trace error
    print('imp error: ' + str(residual))   # imp model error

    elt_size, select_result = selection(ave_fitness, pop_ranked)
    aaa = matingPool(current_gen, select_result)
    bbb = breedPopulation(aaa, elt_size, xoverRate)
    next_gen = mutatePopulation(bbb, generation)
    # for i in range(eltS):
    #   nextGen[i] = AAA[i]   # mutate elite people?
    return residual, error, best_indi, next_gen


def geneticAlgorithm(generations, field_seis):
    imp_pop = initiatePop(popSize, Nsamp)
    progress1 = []
    progress2 = []
    # progress3 = []
    residual = []
    for generation in range(generations):
        print('generation:' + str(generation))
        resi, evolve, best_indi, imp_pop = nextGeneration(imp_pop, generation, field_seis)
        progress1.append(min(evolve))
        progress2.append(np.mean(evolve))
        # progress3.append(best_indi)
        residual.append(resi)
    return imp_pop, progress1, progress2, residual


Imp_best, Min_error, Ave_error, Residual = geneticAlgorithm(Generations, synthetic)


Rc_final = calcuRc(Imp_best)
SYN_final = generateSynthetic(Rc_final, wavelet)
ave_final, error_final, popRanked_final = rankFitness(SYN_final, synthetic)
best_synthetic = SYN_final[popRanked_final[0][0]]
bestModelIndex = popRanked_final[0][0]
bestModel = Imp_best[bestModelIndex]
bestmodel_smoothed = ndi.uniform_filter1d(bestModel, size=3)

plt.plot(best_synthetic, 'r')
plt.plot(synthetic, 'b')
# plt.plot(SyntheticTrace, 'g')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('Inverted and field traces')
plt.show()

plt.plot(np.arange(Generations), Min_error, 'g')
plt.plot(np.arange(Generations), Ave_error, 'b')
# plt.plot(np.arange(generations), np.asarray(residual) * 100, 'black')
plt.ylabel('Error')
plt.xlabel('Generation')
plt.title('Residual between inverted and field traces')
plt.show()

plt.plot(imp)
plt.plot(bestmodel_smoothed)
plt.ylabel('Acoustic Impedance')
plt.xlabel('Time')
plt.title('Inverted and synthetic impedance model')
plt.show()

plt.plot(Residual, 'black')
plt.ylabel('Impedance Model Error')
plt.xlabel('Generation')
plt.title('Inversion precision')
plt.show()
