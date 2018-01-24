import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import random
from PoleChromosome import PoleChromosome
from gen.ThreadAgent import ThreadAgent
from multiprocessing.pool import ThreadPool

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

new_data_with_half_improvement = [0.4666666666666667, 0.3333333333333333, 0.43333333333333335, 0.4, 0.5, 0.4, 0.3, 0.43333333333333335, 0.3333333333333333, 0.3, 0.4, 0.2, 0.3333333333333333, 0.3333333333333333, 0.1, 0.2, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.16666666666666666, 0.26666666666666666, 0.3333333333333333, 0.4, 0.23333333333333334, 0.26666666666666666, 0.43333333333333335, 0.26666666666666666, 0.2, 0.4666666666666667, 0.16666666666666666, 0.26666666666666666, 0.2, 0.4666666666666667, 0.36666666666666664, 0.23333333333333334, 0.43333333333333335, 0.26666666666666666, 0.36666666666666664, 0.4, 0.3333333333333333, 0.26666666666666666]

fig, ax = plt.subplots()
ax.plot(new_data_with_half_improvement[1:32], 'k', label='[0.5, 0.5] probabilities')
ax.set_xlabel('Generation')
ax.set_ylabel('Improvement Rate')
plt.show()