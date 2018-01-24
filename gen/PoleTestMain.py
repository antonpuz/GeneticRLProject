import random

from gen.PoleChromosome import PoleChromosome
from gen.ThreadAgent import ThreadAgent
from multiprocessing.pool import ThreadPool
import numpy as np
import random

random.seed(42)

def print_a_lot(threadID):
    while True:
        print "I am thread {}".format(threadID)

number_of_genomes = 10
ratio_of_new_genoms = 3
number_of_new_gnomes = ratio_of_new_genoms * number_of_genomes
number_of_threads = number_of_genomes + number_of_new_gnomes
pool = ThreadPool(number_of_threads)
epochs = 200
chromosomes = []
for i in range(0,number_of_genomes):
    chromosomes.append(PoleChromosome())

aggregated_results = [30]
improvement_ratio_over_time = []
for i in range(0,epochs):
    genoms = []
    for j in range(0, number_of_genomes):
        genoms.append(chromosomes[j])

    # for j in range(0, number_of_new_gnomes):
    GnomeResults = pool.map(PoleChromosome.staticMutation, np.tile(np.array(chromosomes), (number_of_new_gnomes, 1)))
    for k in range(0, number_of_new_gnomes):
        genoms.append(GnomeResults[k])

    # for j in range(number_of_genomes, number_of_threads):
    #     if random.random() <= mutation_prob:
    #         copiedGen = random.randint(0, number_of_genomes - 1)
    #         to_be_mutated = PoleChromosome(2, poleA=chromosomes[copiedGen].action_matrix)
    #         to_be_mutated.mutate()
    #         genoms.append(to_be_mutated)
    #     else:
    #         poleA = random.randint(0, number_of_genomes - 1)
    #         poleB = random.randint(0, number_of_genomes - 1)
    #         while poleA == poleB:
    #             poleB = random.randint(0, number_of_genomes - 1)
    #         new_pole = PoleChromosome(mode=1, poleA=chromosomes[poleA].action_matrix, poleB=chromosomes[poleB].action_matrix)
    #         genoms.append(new_pole)




    #pool.map(ThreadAgent.calculate_all_in_one, [1, PoleChromosome(), 2, PoleChromosome(), 3, PoleChromosome()])
    print "mapping {}".format(i)
    results = pool.map(ThreadAgent.break_parameters_names, zip(range(0, number_of_threads), genoms))
    print results
    arr = np.array(results)
    comparing_to_parent = np.reshape(arr, [ratio_of_new_genoms+1, number_of_genomes])
    improved_chromosomes = float(sum(sum(comparing_to_parent > comparing_to_parent[0,:])))
    print "improved chromosomes: " + str(improved_chromosomes)
    improvement_ratio = improved_chromosomes/number_of_new_gnomes
    print "improvement ratio: " + str(improvement_ratio)
    improvement_ratio_over_time.append(improvement_ratio)
    print improvement_ratio_over_time
    best_matches = arr.argsort()[-number_of_genomes:][::-1]
    best_gnome = genoms[best_matches[-1]]
    #best_gnome.play()
    for i in range(0, number_of_genomes):
        chromosomes[i] = genoms[best_matches[i]]
    arr.sort()
    best10 = np.take(arr, [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])
    best10mean = np.mean(best10)
    print "average result: " + str(best10mean)
    aggregated_results.append(best10mean)
    print aggregated_results

exit(1)

threads = []
for i in range(0,number_of_threads):
    threadi = ThreadAgent(i)
    threadi.start()
    threads.append(threadi)

for t in threads:
    print t.join()
    print t.get()



