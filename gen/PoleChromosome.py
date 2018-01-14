import random
import numpy
import copy
import gym



class PoleChromosome:
    quantizations = 17
    mutation_prob = 0.0
    vector_normalization_factor = numpy.floor(quantizations/2)
    vector_addition_factor = numpy.floor(quantizations/2)
    number_of_mutations = numpy.floor(numpy.power(quantizations, 4) * 0.05).astype(int) #5% of weights

    def __init__(self, mode=0, poleA = None, poleB = None):
        self.name = "I am a chromosome"
        self.action_matrix = numpy.random.randint(2, size=(PoleChromosome.quantizations, PoleChromosome.quantizations, PoleChromosome.quantizations, PoleChromosome.quantizations))
        if mode==1:
            # print "crossover"
            for feature0 in range(0, PoleChromosome.quantizations):
                for feature1 in range(0, PoleChromosome.quantizations):
                    for feature2 in range(0, PoleChromosome.quantizations):
                        first_matrix = random.randint(0, 1)
                        length_from_mat1 = random.randint(1, PoleChromosome.quantizations - 2)
                        if first_matrix == 0:  # take from poleA
                            self.action_matrix[feature0, feature1, feature2] = copy.deepcopy(poleA[feature0, feature1, feature2])
                        else:  # take from poleB
                            self.action_matrix[feature0, feature1, feature2] = copy.deepcopy(poleB[feature0, feature1, feature2])
        elif mode==2:
            # print "mutating"
            self.action_matrix = copy.deepcopy(poleA)



    def play(self):
        env = gym.make('CartPole-v0')
        observation = env.reset()
        playing_counter = 0
        actions_counter = 0
        while playing_counter < 1:
            action = self.getAction(observation)
            observation, _, done, _ = env.step(action)
            env.render()
            actions_counter = actions_counter + 1
            if done:
                # print "Thread {}: done after {}".format(self.id, actions_counter)
                playing_counter = playing_counter + 1
                observation = env.reset()
                print "Best result: " + str(actions_counter)
                actions_counter = 0








            # for feature in range(0, 5):
            #     first_matrix = random.randint(0,1)
            #     length_from_mat1 = random.randint(1,PoleChromosome.quantizations-2)
            #     if first_matrix==0: #start from poleA
            #         self.action_matrix[feature, 0:length_from_mat1] = poleA[feature, 0:length_from_mat1]
            #         self.action_matrix[feature, length_from_mat1:PoleChromosome.quantizations] = poleB[feature, length_from_mat1:PoleChromosome.quantizations]
            #     else: #start from poleB
            #         self.action_matrix[feature, 0:length_from_mat1] = poleB[feature, 0:length_from_mat1]
            #         self.action_matrix[feature, length_from_mat1:PoleChromosome.quantizations] = poleA[feature, length_from_mat1:PoleChromosome.quantizations]

    def mutate(self):
        for mut in range(0, PoleChromosome.number_of_mutations):
            e = numpy.random.randint(PoleChromosome.quantizations, size=4)
            self.action_matrix[e[0], e[1], e[2], e[3]] = 1 - self.action_matrix[e[0], e[1], e[2], e[3]]

    def getAction(self, observation):
        a = numpy.array(observation)
        #Normalize the observation vector
        a[0] = a[0]/2.0
        a[1] = a[1]/2.0
        a[2] = a[2]/0.21
        a[3] = a[3]/2.0
        a[a>1] = 1
        a[a<-1] = -1
        e = (numpy.round(a*PoleChromosome.vector_normalization_factor)+PoleChromosome.vector_addition_factor).astype(int)
        return self.action_matrix[e[0], e[1], e[2], e[3]]

    def print_something(self):
        print "I am a chromosome"
        return


    @staticmethod
    def staticMutation(genome):
        if random.random() <= PoleChromosome.mutation_prob:
            copiedGen = random.randint(0, len(genome) - 1)
            to_be_mutated = PoleChromosome(2, poleA=genome[copiedGen].action_matrix)
            to_be_mutated.mutate()
            return to_be_mutated
        else:
            poleA = random.randint(0, len(genome) - 1)
            poleB = random.randint(0, len(genome) - 1)
            while poleA == poleB:
                poleB = random.randint(0, len(genome) - 1)
            new_pole = PoleChromosome(mode=1, poleA=genome[poleA].action_matrix, poleB=genome[poleB].action_matrix)
            return new_pole

    @staticmethod
    def break_parameters_names(a):
        return PoleChromosome.staticMutation(a[0], a[1])