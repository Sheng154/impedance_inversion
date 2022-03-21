from typing import List, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import random
# model
tmax = 0.2
tmin = 0
xt = np.arange(0, 200)
# impedance range
max_guess, min_guess = 4500, 1000
# wavelet parameters
f = 50
length = 0.1
dt = 0.001
# Input seismogram
lenSeis = 999 + 1
# Optimisation criterion
L1 = True
# genetic algorithm
popSize = 400
xoverRate = 0.8
mutationRate = 0.4
PopMutateRate = 1
generations = 150
##############################
nsamp = int((tmax - tmin) / dt) + 1
t = []
for i in range(0, nsamp):
    t.append(i * dt)
'''
from SYnthetic_case import *

Synthetic_origin, SyntheticTrace, Synthetic_ctm = SyntheticTrace(Rc, wavelet, nsamp)
wavelet = ricker(f, length, dt)
from SYnthetic_case import imp
'''
from thin_bed import *


# define function of ricker wavelet
def ricker(f, length, dt):
    t0 = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t0 ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t0 ** 2))
    return y


#####################################################################
def createModel(nsamp):
    imp = []
    Rc = []
    for i in range(0, nsamp):
        imp.append(random.random() * (max_guess - min_guess) + min_guess)  # try normal distribution later
    for i in range(0, nsamp - 1):
        Rc.append((imp[i + 1] - imp[i]) / (imp[i + 1] + imp[i]))
    return imp, Rc


def initiatePop(popSize, nsamp):
    imp_pop = []
    for i in range(0, popSize):
        imp = []
        for i in range(0, nsamp):
            # imp.append(random.random() * (max_guess - min_guess) + min_guess)
            # imp.append(np.random.normal(2200, 500))  # Determines the range of synthetic impedance model
            imp.append(np.random.uniform(1000, 3500))
            # imp_smooth = ndi.uniform_filter1d(imp, size=3)
        imp_pop.append(imp)
    return imp_pop


def calcurc(imp):
    Rc_pop = []
    nmodel = np.shape(imp)[0]
    nsamp = np.shape(imp)[1]
    for i in range(nmodel):
        Rc = []
        imp_individule = imp[i]
        for i in range(0, nsamp - 1):
            Rc.append((imp_individule[i + 1] - imp_individule[i]) / (imp_individule[i + 1] + imp_individule[i]))
        Rc_pop.append(Rc)
    return Rc_pop


def generateSynthetic(Rc_pop, wavelet):
    synthetic_pop = []
    for i in range(np.shape(Rc_pop)[0]):
        trace = np.convolve(Rc_pop[i], wavelet, mode='same')
        trace_norm = trace / max(trace)
        synthetic_pop.append(trace_norm)
    return synthetic_pop  # normalised synthetic trace population


def rankFitness(synthetic_norm, trace_norm):
    fitness = {}
    error = []
    error1 = []
    synthetic_norm = np.asarray(synthetic_norm)
    trace_norm = np.asarray(trace_norm)
    for i in range(synthetic_norm.shape[0]):
        err_ = 0
        diff = synthetic_norm[i] - trace_norm
        for j in range(len(diff)):
            err_ = err_ + abs(diff[j] / trace_norm[j])
        error1.append(err_ / len(diff))  # 每条trace的error，计算方式：average percentage error per sample
        error.append(sum(abs(diff)))  # E(m) L1-norm, 计算方式：每条trace的总error
        # error[i] = sum(np.square(diff))  # L2-norm
        fitness[i] = 1 / error[i]
    error_total = sum(error)  # numerator  这一代所有人的error
    for keys in fitness:
        fitness[keys] = fitness[keys] * error_total
    fit = fitness.values()
    ave_fitness = sum(fit) / popSize
    return ave_fitness, error, sorted(fitness.items(), key=lambda item: item[1], reverse=True)


def selection(ave, popRanked):
    selectionResults = []
    EliteSize = 0
    for i in range(len(popRanked)):
        if popRanked[i][1] > ave:
            EliteSize = EliteSize + 1
            selectionResults.append(popRanked[i][0])  # better than average are carried forward
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_per'] = 100 * df.cum_sum / df.Fitness.sum()
    for i in range(popSize - EliteSize):  # 需要补足的个体数
        pick = 100 * random.random()
        for i in range(len(popRanked)):
            if pick <= df.iat[i, 3]:  # 从前往后依次挑选
                selectionResults.append(popRanked[i][0])
                break
    return EliteSize, selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    childP1 = []
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startGene = min(geneA, geneB)  # 双点cross-over
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    parent2 = list(parent2)
    child = parent2[0: startGene] + childP1 + parent2[endGene: len(parent2)]
    return child


def breedPopulation(matingpool, eliteSize, xoverRate):
    children = []
    pool = random.sample(matingpool, len(matingpool))  # 打乱impedance model pool
    for i in range(eliteSize):
        children.append(matingpool[i])
    for i in range(len(matingpool) - eliteSize):
        if random.random() < xoverRate:
            child = breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        else:
            pool = random.sample(matingpool, len(matingpool))
            children.append(pool[i])
    return children


def mutatePopulation(population, mutationRate):
    mutatedPop = []
    # plt.plot(BBB[1])
    for i in range(len(population)):
        indiv = population[i]
        if random.random() < (1 - PopMutateRate):  # populated into next generation without mutation
            mutatedPop.append(indiv)
            continue
        for j in range(len(indiv)):  # mutate according to mutateRate
            if random.random() < mutationRate:
                indiv[j] = np.random.normal(2200, 300)
        mutatedPop.append(indiv)
    # plt.plot(mutatedPop[1])
    # plt.show()
    return mutatedPop


def nextGeneration(currentGen, mutationRate, xoverRate, fieldSeis):
    #currentGen = []
   # for items in currentGen_raw:
       # currentGen.append(ndi.uniform_filter1d(items, size=3))
    Rc_pop = calcurc(currentGen)
    SYN = generateSynthetic(Rc_pop, wavelet)
    ave, error, popRanked = rankFitness(SYN, fieldSeis)
    best_curGen = currentGen[popRanked[0][0]]
    imp_residual = best_curGen - imp
    err_ = 0
    for j in range(len(imp_residual)):
        err_ = err_ + abs(imp_residual[j] / imp[j])
    error1 = err_ / len(imp_residual)  # 每个model的error，计算方式：average percentage error per sample 1/2
    residual = sum(abs(imp_residual)) / sum(imp)  # 两种误差计算方式2/2
    # min_error = min(error)
    # ave_error = np.mean(error)
    print('trace error: ' + str(min(error)))
    print('imp error: ' + str(error1))
    # print('imp error: ' + str(residual))
    eltS, selR = selection(ave, popRanked)
    AAA = matingPool(currentGen, selR)
    BBB = breedPopulation(AAA, eltS, xoverRate)
    nextGen = mutatePopulation(BBB, mutationRate)
    # for i in range(eltS):
    #   nextGen[i] = AAA[i]   # mutate elite people?
    return residual, error1, error, best_curGen, nextGen


def geneticAlgorithm(popSize, mutationRate, xoverRate, generations, fieldSeis):
    imp_pop = initiatePop(popSize, nsamp)
    progress1 = []
    progress2 = []
    progress3 = []
    residual = []
    residual1 = []
    for i in range(generations):
        print(i)
        # if i < 0.25 * generations:
        #   resi, resi1, evolve, best_curGen, imp_pop = nextGeneration(imp_pop, mutationRate, xoverRate, fieldSeis)
        # else:
        mutationRate = mutationRate - i * (0.5 * mutationRate / generations)
        resi, resi1, evolve, best_curGen, imp_pop = nextGeneration(imp_pop, mutationRate, xoverRate, fieldSeis)
        # print('optimal shouyuansu'+ str(imp_pop[0][0]))
        progress1.append(min(evolve))
        progress2.append(np.mean(evolve))
        progress3.append(best_curGen)
        residual.append(resi)
        residual1.append(resi1)

    # plt.plot(residual)
    # plt.ylabel('Residual(%)')
    # plt.xlabel('Generation')
    # plt.show()
    return imp_pop, progress1, progress2, progress3, residual, residual1


imp_POP, min_error, ave_error, imp_models, residual, residual1 = geneticAlgorithm(popSize, mutationRate, xoverRate,
                                                                                  generations, synthetic)

Rc_final = calcurc(imp_POP)
SYN_final = generateSynthetic(Rc_final, wavelet)
ave_final, error_final, popRanked_final = rankFitness(SYN_final, Synthetic_ctm)
best_synthetic = SYN_final[popRanked_final[0][0]]
bestModelIndex = popRanked_final[0][0]
bestModel = imp_POP[bestModelIndex]
bestmodel_smoothed = ndi.uniform_filter1d(bestModel, size=3)

indices = [int(generations - 1)]
checkpoint = [imp_models[index] for index in indices]
rc_check = calcurc(checkpoint)
for items in rc_check:
    plt.plot(items, label='final')
plt.plot(Rc, label='synthetic')
plt.legend(loc="upper left")
plt.show()
power_fcheck = []
power_pcheck = []
for items in rc_check:
    f_Rcheck, p_Rcheck = power(items)
    power_fcheck.append(f_Rcheck)
    power_pcheck.append(p_Rcheck)
for i in range(len(indices)):
    plt.plot(power_fcheck[i], power_pcheck[i])
    # plt.legend(loc="upper left")
# f_Rc, p_Rc = power(Rc)
# plt.plot(f_Rc, p_Rc, 'purple')
plt.ylabel('Rc')
plt.show()
# plt.plot(checkpoint[0])
# plt.plot(checkpoint[1])
# plt.plot(checkpoint[2])
# plt.show()

plt.plot(best_synthetic, 'r')
plt.plot(Synthetic_ctm, 'b')
plt.plot(SyntheticTrace, 'g')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()

plt.plot(np.arange(generations), min_error, 'g')
plt.plot(np.arange(generations), ave_error, 'b')
# plt.plot(np.arange(generations), np.asarray(residual) * 100, 'black')
plt.ylabel('Error')
plt.xlabel('Generation')
plt.show()

plt.plot(imp)
plt.plot(bestmodel_smoothed)
plt.ylabel('Acoustic Impedance')
plt.xlabel('Time')
plt.show()

plt.plot(residual, 'black')
plt.plot(residual1, 'r')
plt.ylabel('Impedance Model Error')
plt.xlabel('Generation')
plt.show()


def power(timeseries):
    fourier_transform = np.fft.rfft(timeseries)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, 1 / (2 * dt), len(power_spectrum))
    return frequency, power_spectrum

# SYN_best = generateSynthetic(Rc_best, wavelet)
# print('SYN_best shape' + str(np.shape(SYN_best)))
# plt.plot(SYN_best, 'g');
# plt.plot(Synthetic_ctm, 'b');
# plt.plot(SyntheticTrace, 'r')
