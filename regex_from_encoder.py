#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import csv

def elem_class(elem):
    return type(elem).__name__
def create_hot_vector(hot_index, length):
    res = np.zeros(length, int)
    res[hot_index] = 1
    return res

with open('resources/regextrain.csv') as f:
    reader = csv.reader(f, delimiter=',')
    l = [x.split(",") for x in f]
    np_l = map(lambda t : np.array(t), l)
    np_ints = map(lambda arr_e : map(lambda element : int(element[0:1]), arr_e), np_l)

    #Make hot vectors out of each entry
    #hot_vectors = map(lambda arr_e : np.array(map(lambda element : create_hot_vector(element, 10), arr_e)).flatten(), np_ints)

    #Start the NN part
    x = tf.placeholder(tf.float32, [None, 30])
    W1 = tf.Variable(tf.random_normal([30, 30]))
    b1 = tf.Variable(tf.random_normal([30]))
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal([30, 30]))
    b2 = tf.Variable(tf.random_normal([30]))
    y_not_normalized = tf.nn.relu(tf.matmul(y1, W2) + b2)

    #parts_of_y = tf.split(y_not_normalized, 10, 1)
    #argmax_of_y = map(lambda a : tf.argmax(a), parts_of_y)

    y = y_not_normalized

    y_ = tf.placeholder(tf.float32, [None, 30])

    cross_entropy = tf.reduce_mean(
        #tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
        tf.multiply(tf.squared_difference(y_, y), 1))
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for epoch in range(1500):
        avg_cost = 0.
        total_batch = int(len(np_ints) / 10)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = np_ints[i*10 : (i+1)*10]
            batch_y = batch_x

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

    # Test trained model
    mse = tf.reduce_mean(tf.squared_difference(y, y_))

    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(mse, tf.float32))
    print(sess.run(accuracy, feed_dict={x: np_ints,
                                        y_: np_ints}))


    output = sess.run([y], feed_dict={x: np_ints, y_: np_ints})
    #print output[0]

    print elem_class(y)

    count = 0
    for a in output:
        for b in a:
            if count < 3:
                count += 1
                print len(b)
                print b



