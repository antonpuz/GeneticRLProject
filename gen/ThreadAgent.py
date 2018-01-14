import threading
import gym
from gen.PoleChromosome import PoleChromosome
import numpy
from multiprocessing.pool import ThreadPool

class ThreadAgent (threading.Thread):

    number_of_executions = 40
    executionPool = ThreadPool(number_of_executions)

    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.id = threadID
        self.env = gym.make('CartPole-v0')
        self.observation = self.env.reset()
        self.done = 0
        self.agent = PoleChromosome()
        self.max_result = 0

    def run(self):
        print "Thread {} started".format(self.id)
        playing_counter = 0
        actions_counter = 0
        self.observation = self.env.reset()
        while playing_counter < 5:
            #action = 0 if self.observation[2] < 0 else 1  # 0 - left, 1 - right; Negative angle = pole on left side
            action = self.agent.getAction(self.observation)
            # action = 1 # 0 - left, 1 - right; Negative angle = pole on left side
            self.observation, _, self.done, _ = self.env.step(action)
            # print observation
            # print self.observation[3]
            # self.env.render()
            # time.sleep(1)
            actions_counter = actions_counter + 1
            if self.done:
                # print "Thread {}: done after {}".format(self.id, actions_counter)
                self.max_result = max(self.max_result, actions_counter)
                playing_counter = playing_counter + 1
                self.observation = self.env.reset()
                actions_counter = 0


    def get(self):
        return self.max_result

    @staticmethod
    def calculateOneGameForGnome(genome):
        env = gym.make('CartPole-v0')
        actions_counter = 0
        observation = env.reset()
        done = False
        while done != True:

            action = genome.getAction(observation)
            actions_counter = actions_counter + 1
            # action = 1 # 0 - left, 1 - right; Negative angle = pole on left side
            observation, _, done, _ = env.step(action)
            # print "action taken" + str(done)
            # print observation
            # print self.observation[3]
            # self.env.render()
            # time.sleep(1)

        # print "last observation: " + str(observation)
        return actions_counter








    @staticmethod
    def calculate_all_in_one_parallelized(genome):
        results = ThreadAgent.executionPool.map(ThreadAgent.calculateOneGameForGnome, numpy.repeat(genome, ThreadAgent.number_of_executions))
        # print results
        return numpy.mean(results)



    @staticmethod
    def calculate_all_in_one(threadID, genome):
        # if done:
        #     # print "Thread {}: done after {}".format(threadID, actions_counter)
        #     playing_counter = playing_counter + 1
        #     observation = env.reset()
        #     print actions_counter
        #     return actions_counter
        # results = ThreadAgent.executionPool.map(ThreadAgent.calculateOneGameForGnome, numpy.repeat(genome, ThreadAgent.number_of_executions))
        # return numpy.mean(results)
        return ThreadAgent.calculate_all_in_one_parallelized(genome)

    @staticmethod
    def break_parameters_names(a):
        return ThreadAgent.calculate_all_in_one(a[0], a[1])