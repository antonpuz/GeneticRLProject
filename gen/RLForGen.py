import numpy as np
import scipy as sp
import tensorflow as tf

from gen import PoleChromosome

def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer

# H = 8 # number of hidden layer neurons
# learning_rate = 1e-2
# gamma = 0.99 # discount factor for reward
# decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
# resume = False # resume from previous checkpoint?
#
# bs = 3 # Batch size
# input_size = PoleChromosome.quantizations*PoleChromosome.quantizations*PoleChromosome.quantizations*PoleChromosome.quantizations # input dimensionalitys
#
tf.reset_default_graph()
#
# observations = tf.placeholder(tf.float32, [None,input_size] , name="input_x")
# W1 = tf.get_variable("W1", shape=[input_size, H],
#            initializer=tf.contrib.layers.xavier_initializer())
# layer1 = tf.nn.relu(tf.matmul(observations,W1))
# W2 = tf.get_variable("W2", shape=[H, 1],
#            initializer=tf.contrib.layers.xavier_initializer())
# score = tf.matmul(layer1,W2)
# probability = tf.nn.sigmoid(score)
#
# tvars = tf.trainable_variables()
# input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
# advantages = tf.placeholder(tf.float32,name="reward_signal")
# adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
# W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
# W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
# batchGrad = [W1Grad,W2Grad]
# loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
# loss = -tf.reduce_mean(loglik * advantages)
# newGrads = tf.gradients(loss,tvars)
# updateGrads = adam.apply_gradients(zip(batchGrad,tvars))
#
# mH = 256 # model layer size

init = tf.initialize_all_variables()

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    gradBuffer = sess.run(tvars)
    gradBuffer = resetGradBuffer(gradBuffer)

    while episode_number <= 5000:
        # print observation
        #vfunc = np.vectorize(lambda t: sigmoid(t))
        #print vfunc(observation)-0.5
        #print np.multiply(np.power(np.abs(observation), 1.0/5), np.sign(observation))

        x = np.reshape(observation, [1, matrix.size])

        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        # action = 0 if observation[2] < 0 else 1

        # record various intermediates (needed later for backprop)
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        observation, reward, done, info = env.step(action)

        reward_sum += reward

        ds.append(done * 1)
        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            print "done after {}".format(reward_sum)
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs, drs, ys, ds = [], [], [], []  # reset array memory

            discounted_epr = discount_rewards(epr).astype('float32')
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})

            # If gradients becom too large, end training process
            if np.sum(tGrad[0] == tGrad[0]) == 0:
                break
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if switch_point + batch_size == episode_number:
                switch_point = episode_number

                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                gradBuffer = resetGradBuffer(gradBuffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                if drawFromModel == False:
                    print 'World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % (
                    real_episodes, reward_sum / real_bs, action, running_reward / real_bs)
                    if reward_sum / batch_size > 200:
                        break
                reward_sum = 0

            observation = env.reset()
            batch_size = real_bs


