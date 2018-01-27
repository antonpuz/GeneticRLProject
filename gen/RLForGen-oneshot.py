import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import random
from PoleChromosome import PoleChromosome
from gen.ThreadAgent import ThreadAgent
from multiprocessing.pool import ThreadPool
import time

env = gym.make('CartPole-v0')

random.seed(42)

input_size = PoleChromosome.quantizations*PoleChromosome.quantizations*PoleChromosome.quantizations*PoleChromosome.quantizations # input dimensionalitys
number_of_genomes = 10
ratio_of_new_genoms = 3
number_of_new_gnomes = ratio_of_new_genoms * number_of_genomes
number_of_threads = number_of_genomes + number_of_new_gnomes
chromosomes = []

for i in range(0,number_of_genomes):
    chromosomes.append(PoleChromosome())

pool = ThreadPool(number_of_genomes)
genoms = chromosomes
gamma = 0.95
alpha = 0.05

total_episodes = 5000  # Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 1

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    # (lr=1e-4, s_size=input_size, a_size=2, h_size=40)
    def __init__(self, lr, s_size, a_size, h_size):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        self.hidden1 = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(self.hidden1, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        # self.loss = tf.reduce_mean(tf.abs(-self.responsible_outputs + self.reward_holder))
        self.loss = tf.reduce_mean(tf.pow(-self.responsible_outputs + self.reward_holder, 2))
        # self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
        # self.loss = tf.reduce_mean(self.responsible_outputs + self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


tf.reset_default_graph()  # Clear the Tensorflow graph.

myAgent = agent(lr=1e-4, s_size=input_size, a_size=3, h_size=4)  # Load the agent.

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0

    total_scores = [30]
    total_improvement_rate = []
    total_transformation_prob = np.array([])
    start = time.time()
    total_execution_time = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0



    while i < total_episodes:
        results = []
        genoms = []
        trans_prob_array = []
        success_rate_array = []
        for chrom in range(0, number_of_genomes):
            genoms.append(chromosomes[chrom])
            results.append(ThreadAgent.calculate_all_in_one_parallelized(chromosomes[chrom]))

        # results = pool.map(ThreadAgent.break_parameters_names, zip(range(0, number_of_genomes), chromosomes))

        for newGenome in range (0, number_of_new_gnomes):
            current_chromosome_id = newGenome % number_of_genomes
            current_chromosome = chromosomes[current_chromosome_id]


            ep_history = []
            new_pole = current_chromosome
            reward_history = []
            gnome_history = []
            for evolution_number in range(0, update_frequency):

                # Probabilistically pick an action given our network outputs.
                a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [np.ndarray.flatten(new_pole.action_matrix)]})
                trans_prob_array.append(a_dist)
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)

                if random.random() <= alpha and a<2:
                    a = 1-a

                if a==0:
                    copiedGen = random.randint(0, number_of_genomes - 1)
                    new_pole = PoleChromosome(2, poleA=new_pole.action_matrix)
                    new_pole.mutate()
                elif a==1:
                    poleB = random.randint(0, number_of_genomes - 1)
                    while current_chromosome_id == poleB:
                        poleB = random.randint(0, number_of_genomes - 1)
                    new_pole = PoleChromosome(mode=1, poleA=new_pole.action_matrix, poleB=chromosomes[poleB].action_matrix)
                else: #random opearion
                    if random.random() <= 0.5:
                        new_pole = PoleChromosome(2, poleA=new_pole.action_matrix)
                        new_pole.mutate()
                    else:
                        poleB = random.randint(0, number_of_genomes - 1)
                        while current_chromosome_id == poleB:
                            poleB = random.randint(0, number_of_genomes - 1)
                        new_pole = PoleChromosome(mode=1, poleA=new_pole.action_matrix,
                                                  poleB=chromosomes[poleB].action_matrix)


                r = ThreadAgent.calculate_all_in_one_parallelized(new_pole)
                reward_compared_to_base = -0.1
                if a==2:
                    reward_compared_to_base = -0.05
                elif results[current_chromosome_id] < r:
                    reward_compared_to_base=1

                ep_history.append([np.ndarray.flatten(new_pole.action_matrix), a, reward_compared_to_base, 0])

                reward_history.append(r)
                gnome_history.append(new_pole)

            # print ep_history
            # print a_dist
            success_rate_array.append(np.array(ep_history)[:, 2].mean())

            ep_history = np.array(ep_history)
            # ep_history[:, 2] = discount_rewards(ep_history[:, 2])
            feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                         myAgent.action_holder: ep_history[:, 1], myAgent.state_in: np.vstack(ep_history[:, 0])}

            # print "outputs " + str(sess.run(myAgent.output, feed_dict=feed_dict))
            # print "reward " + str(sess.run(myAgent.reward_holder, feed_dict=feed_dict))
            # print "actions " + str(sess.run(myAgent.action_holder, feed_dict=feed_dict))
            # print "indexes " + str(sess.run(myAgent.indexes, feed_dict=feed_dict))
            # print "responsible_outputs " + str(sess.run(myAgent.responsible_outputs, feed_dict=feed_dict))
            # print "gradients " + str(sess.run(myAgent.gradients, feed_dict=feed_dict))
            # print "loss " + str(sess.run(myAgent.loss, feed_dict=feed_dict))

            grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
            _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

            reward_np_arr = np.array(reward_history)
            best_match = reward_np_arr.argsort()[-1:][::-1][0]

            results.append(reward_history[best_match])
            genoms.append(gnome_history[best_match])

        total_transformation_prob = np.append(total_transformation_prob, np.array(trans_prob_array).mean(0))

        print results
        arr = np.array(results)
        best_matches = arr.argsort()[-number_of_genomes:][::-1]
        print best_matches

        improvement_rate = np.array(success_rate_array).mean()
        total_improvement_rate.append(improvement_rate)

        # best_gnome = genoms[best_matches[-1]]
        # best_gnome.play()
        for best_gnome in range(0, number_of_genomes):
            chromosomes[best_gnome] = genoms[best_matches[best_gnome]]
        arr.sort()
        best10 = np.take(arr, [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
        total_scores.append(np.mean(best10))
        print "average result after step: " + str(i+1)
        print total_scores
        print "average transformations probabilities:"
        print repr(total_transformation_prob)
        print "Improvement rate: "
        print total_improvement_rate
        end = time.time()
        print "Execution time: "
        total_execution_time.append(end - start)
        print total_execution_time


        i += update_frequency

