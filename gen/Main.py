import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import random
from PoleChromosome import PoleChromosome
from gen.ThreadAgent import ThreadAgent
from multiprocessing.pool import ThreadPool

env = gym.make('CartPole-v0')

random.seed(42)


def randomAgent():
    number_of_genomes = 10
    chromosomes = []
    results = []
    for i in range(0, number_of_genomes):
        chromosomes.append(PoleChromosome())
        results.append(ThreadAgent.calculate_all_in_one_parallelized(chromosomes[i]))

    print results
    arr = np.array(results)
    print np.mean(arr)


def fixingAgent():
    playing_counter = 0
    actions_counter = 0
    results = []
    while playing_counter < 1000:
        observation = env.reset()
        done = False
        while done!=True:
            action = 0 if observation[2] < 0 else 1  # 0 - left, 1 - right; Negative angle = pole on left side
            observation, _, done, _ = env.step(action)
            env.render()
            actions_counter += 1

        results.append(actions_counter)
        actions_counter = 0
        playing_counter += 1

    # print results
    arr = np.array(results)
    print np.mean(arr)
    exit(1)

def fixedGenetic():
    print "here"





fixingAgent()
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Make some fake data.
a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]
e = [1,2,3,4,5,6,7,8,9]
genetic_with_half = [30, 48.704999999999998, 60.81750000000001, 73.465000000000003, 75.517500000000013, 81.394999999999996, 89.135000000000005, 108.255, 111.27000000000001, 113.75750000000001, 121.09750000000001, 117.2325, 123.22999999999999, 126.10249999999999, 126.52000000000001, 125.30500000000002, 130.655, 133.33750000000001, 140.11750000000001, 141.41249999999999, 151.23999999999998, 150.57749999999999, 158.27249999999998, 160.02250000000001, 163.33750000000001, 159.91500000000002, 162.72749999999999, 162.32249999999999, 166.7175, 169.15000000000001, 171.2825]
#genetic_with_half = [30, 48.704999999999998, 60.81750000000001, 73.465000000000003, 75.517500000000013, 81.394999999999996, 89.135000000000005, 108.255, 111.27000000000001, 113.75750000000001, 121.09750000000001, 117.2325, 123.22999999999999, 126.10249999999999, 126.52000000000001, 125.30500000000002, 130.655, 133.33750000000001, 140.11750000000001, 141.41249999999999, 151.23999999999998, 150.57749999999999, 158.27249999999998, 160.02250000000001, 163.33750000000001, 159.91500000000002, 162.72749999999999, 162.32249999999999, 166.7175, 169.15000000000001, 171.2825, 169.91500000000002, 175.41499999999999, 177.69499999999999, 178.4325, 181.01999999999998, 183.5675, 186.93750000000003, 186.9675, 186.62, 185.315, 189.13499999999999]
genetic_only_mutation = [30, 42.8125, 54.322500000000005, 60.807500000000005, 65.060000000000002, 70.719999999999999, 73.185000000000002, 79.585000000000008, 81.857500000000002, 88.342500000000001, 91.690000000000012, 97.215000000000003, 97.362500000000011, 102.42750000000001, 102.39250000000001, 104.54749999999999, 103.38250000000001, 104.405, 107.06500000000001, 111.785, 112.23499999999999, 115.5275, 118.22, 122.32000000000001, 120.33, 122.6825, 123.61749999999999, 125.0625, 133.57250000000002, 134.09999999999999, 138.17749999999998]
genetic_only_crossover = [30, 39.102500000000006, 43.399999999999991, 51.297499999999999, 58.802499999999995, 71.164999999999992, 79.807500000000005, 90.922499999999999, 96.224999999999994, 99.242500000000007, 106.12249999999999, 108.82250000000002, 123.68250000000003, 123.96250000000002, 127.13499999999999, 129.71000000000001, 128.49250000000001, 130.22750000000002, 130.93750000000003, 134.96250000000001, 135.2475, 132.55000000000001, 134.09499999999997, 137.35250000000002, 135.08999999999997, 136.84249999999997, 138.58249999999998, 138.24499999999998, 135.85749999999999, 133.07249999999999, 139.76500000000001]
# Create plots with pre-defined labels.
fig, ax = plt.subplots()
print len(genetic_with_half)
print len(genetic_only_mutation)
print len(genetic_only_crossover)
ax.plot(genetic_with_half, 'k', label='[0.5, 0.5] probabilities')
ax.plot(genetic_only_mutation, 'k--', label='only mutation')
ax.plot(genetic_only_crossover, 'k:', label='only crossover')
# ax.set_title('Genetic algorithm with [0.5,0.5] probabilities')
ax.set_xlabel('generation')
ax.set_ylabel('actions')
legend = ax.legend(loc='upper left', shadow=True)#, fontsize='x-large')

# Put a nicer background color on the legend.
# legend.get_frame().set_facecolor('#00FFCC')

plt.show()