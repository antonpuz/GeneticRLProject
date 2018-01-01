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
epochs = 20
chromosomes = []
for i in range(0,number_of_genomes):
    chromosomes.append(PoleChromosome())

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
    best_matches = arr.argsort()[-number_of_genomes:][::-1]
    best_gnome = genoms[best_matches[-1]]
    #best_gnome.play()
    for i in range(0, number_of_genomes):
        chromosomes[i] = genoms[best_matches[i]]
    arr.sort()
    best10 = np.take(arr, [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])
    print "average result: " + str(np.mean(best10))

exit(1)

threads = []
for i in range(0,number_of_threads):
    threadi = ThreadAgent(i)
    threadi.start()
    threads.append(threadi)

for t in threads:
    print t.join()
    print t.get()



