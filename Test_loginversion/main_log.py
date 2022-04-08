# 20220323
import random
import scipy.ndimage as ndi
from deap import base
from deap import creator
from deap import tools
from functions import *
from SYnthetic_case import mback, imp_log, wavelet, omtx
# from thin_bed import imps, model_trace, wavelet
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.metrics import r2_score
import pylops
import scipy

# wavelet parameters
f = 500
length = 0.071
dt = 1e-3  # 1ms, 1,000Hz
# Optimisation criterion
L1 = True
###########################
np.random.seed(12)

# Parameters
tmax = 200 * 1e-3  # 10ms
tmin = 0
nsamp = int((tmax - tmin) / dt)
xt = np.arange(0, 200)
pop_no = 1000
generations = 200
g = 0
CXPB = 0.7
MUTPB = 0.3
imp_seabed = imp_log[0]
t_samples = 200


# Set up GA operators
creator.create('FitnessMulti', base.Fitness, weights=(1.0, ))
creator.create('Individual', list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("prior_range", np.random.uniform, 7.3, 8)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.prior_range, t_samples)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register('select1', tools.selBest)
toolbox.register("select2", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mate2", tools.cxUniform)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.05)
# toolbox.register("mutate", tools.mutUniformInt, low=7.4, up=7.9, indpb=0.05)
stats_fit = tools.Statistics(lambda indi: ind.fitness.values)
stats_fit.register("avg", np.mean)
stats_fit.register("std", np.std)
stats_fit.register("min", np.min)
stats_fit.register("max", np.max)

imp_pop = toolbox.population(n=pop_no)
imp_pop[2] = filtfilt(np.ones(3) / float(int(3)), 1, imp_pop[1])
print(imp_pop[1].values)
plt.plot(imp_pop[2])
plt.plot(imp_pop[1])
plt.show()
# imp_pop = filtfilt(np.ones(5) / float(5), 1, ind) for ind in imp_pop
Error = 10
evolution1 = []
evolution1_min = []
evolution1_max = []
evolution1_ave = []
evolution2 = []
evolution3 = []
fit_trace = []
fit_model = []
err_imp_result = np.empty([generations, 3])

while Error > 0 and g < generations:
    g = g + 1
    print("-- Generation %i --" % g)
    # SELECTION
    offspring1 = toolbox.select1(imp_pop, int(pop_no / 2))  # SelBest
    offspring2 = toolbox.select2(imp_pop, int(pop_no / 2))  # selTournament
    offspring = offspring1 + offspring2
    offspring = list(map(toolbox.clone, offspring))

    # CROSSOVER
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            # toolbox.mate1(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    # offspring = stepped(offspring)

    # MUTATION
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    # offspring = stepped(offspring)

    # Evaluate the fitness of individuals
    syn = SyntheticConvmtx(wavelet, offspring)
    for i in range(pop_no):
        r2_trace = scipy.stats.linregress(syn[i], mtrace)
        r2_model = scipy.stats.linregress(offspring[i], imp_log)
    fit_trace.append(r2_trace.rvalue ** 2)
    fit_model.append(r2_model.rvalue ** 2)
    rc_pop = calcuRc(offspring)
    fitnesses = map(toolbox.evaluate, syn, rc_pop)  # 求取更新个体的fitness值
    for ind, fit in zip(offspring, fitnesses):  # 重新分配fitness值
        ind.fitness.values = fit

    imp_pop[:] = offspring

    fits = [ind.fitness.values[0] for ind in imp_pop]  # trace error absolute value
    # spiking = [1 / ind.fitness.values[1] for ind in imp_pop]
    hof = imp_pop[fits.index(max(fits))]     # trace误差最小的model的residual，而非真正的最佳model
    hof = filtfilt(np.ones(3) / float(3), 1, hof)

    length = len(imp_pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5
    A = 1 / min(fits)
    B = 1 / max(fits)
    C = 1 / mean
    Error = 100 * B / sum(abs(mtrace))  # percentage error of synthetic traces
    print("  Min %s" % B)  # 以下三个是trace的绝对误差
    print("  Max %s" % A)
    print("  Avg %s" % C)
    print("  Std %s" % std)
    print('  Err %s' % Error)  # average percentage error of synthetic and field traces 百分误差, not the same as Vardy
    evolution1.append(Error)
    evolution1_min.append(B)
    evolution1_max.append(A)
    evolution1_ave.append(mean)
    # evolution2.append(np.mean(spiking))

    err_imp = []
    for ind in offspring:
        diff_imp = []
        for k in range(len(ind)):
            diff_imp.append(abs((ind[k] - imp_log[k]) / imp_log[k]))
        err_imp.append(sum(diff_imp) / len(diff_imp))  # percentage model residual as Vardy
    err_imp_result[g - 1, 0] = min(err_imp)
    err_imp_result[g - 1, 1] = max(err_imp)
    err_imp_result[g - 1, 2] = np.mean(err_imp)

    eval1 = scipy.stats.linregress(hof, imp_log)
    print('r2 for impedance: ' + str(eval1.rvalue ** 2))
    print('r2 for trace: ' + str(max(fit_trace)))
    evolution2.append(max(fit_trace))
    evolution3.append(max(fit_model))

best_ind = np.asarray(imp_pop[fits.index(max(fits))])
best_ind_lp = butter_lowpass_filter(best_ind, 300, 1000)

# best_whole_smoothed = ndi.uniform_filter1d(best_ind_whole, size=3)
best_syn = omtx * best_ind
r2_trace = scipy.stats.linregress(best_syn, mtrace)
print('r2_trace:' + str(r2_trace.rvalue ** 2))

minv = pylops.avo.poststack.PoststackInversion(mtrace, wavelet / 2, m0=mback, explicit=False, simultaneous=False)[0]

plt.plot(evolution3)
plt.show()

plot_x = np.arange(1, generations + 1)
plt.plot(plot_x, err_imp_result[:, 0], label='min', linewidth=4)
plt.plot(plot_x, err_imp_result[:, 1], label='max', linewidth=4)
plt.plot(plot_x, err_imp_result[:, 2], label='mean', linewidth=4)
plt.xlabel('Generations')
plt.ylabel('Impedance error')
plt.legend()
plt.show()

plot_x = np.arange(1, t_samples + 1)
plt.subplot(2, 1, 1)
plt.plot(evolution1)
plt.xlabel('Generations')
plt.ylabel('Trace error')
plt.subplot(2, 1, 2)
plt.plot(evolution2)
plt.xlabel('Generations')
plt.ylabel('r2_trace')
plt.show()

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].spines['bottom'].set_linewidth(4)
axs[0].spines['top'].set_linewidth(4)
axs[0].spines['left'].set_linewidth(4)
axs[0].spines['right'].set_linewidth(4)
axs[0].plot(plot_x, best_syn, label='best_syn', linewidth=4)
axs[0].plot(plot_x, mtrace, label='mtrace', linewidth=4)
axs[0].set_xlim(0, 200)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
axs[0].set_ylabel('Normalised amplitude', fontsize=24)
axs[0].tick_params(direction='out', length=15, width=4, grid_color='r', grid_alpha=0.5)
axs[0].legend()

axs[1].plot(plot_x, best_ind, label='best_ind', linewidth=4)
axs[1].plot(plot_x, imp_log, label='best_ind', linewidth=4)
axs[1].set_xlim(0, 200)
axs[1].set_ylim(7.3, 8)
axs[1].set_xlabel('Time (ms)', fontsize=24)
axs[1].set_ylabel('Impedance' '$\mathregular{(m/s · g/cm^{3})}$', fontsize=24)
axs[1].spines['bottom'].set_linewidth(4)
axs[1].spines['top'].set_linewidth(4)
axs[1].spines['left'].set_linewidth(4)
axs[1].spines['right'].set_linewidth(4)
axs[1].tick_params(direction='out', length=15, width=4, grid_color='r', grid_alpha=0.5)
axs[1].legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

####################################
fig, ax = plt.subplots()
ax.plot(plot_x, imp_log, label='Synthetic impedance model', linewidth=4)
ax.plot(plot_x, best_ind, label='Inverted impedance model', linewidth=4)
ax.tick_params(direction='out', length=15, width=4, grid_color='r', grid_alpha=0.5)
# ax.spines[bottom].set_linewidth(size).
mpl.rcParams['axes.linewidth'] = 2  # set the value globally
ax.set_xlim(0, 200)
ax.set_ylim(7.3, 8)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax.spines['bottom'].set_linewidth(4)
ax.spines['top'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.set_xlabel('Time (ms)', fontsize=24)
ax.set_ylabel('Impedance' '$\mathregular{(m/s · g/cm^{3})}$', fontsize=24)
ax.legend(fontsize=32)
# plt.plot(imps, 'r')
# plt.plot(imps_, 'b')
plt.show()