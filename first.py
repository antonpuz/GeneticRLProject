import tensorflow as tf

print "Hello world!"

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
a = tf.Variable([1.0])
linear_model = W*x + b*a

init = tf.global_variables_initializer()
sess.run(init)

print (linear_model.eval({x: [1, 2, 3, 4]}))

diff = tf.square(linear_model - y)
loss = tf.reduce_sum(diff)

print(loss.eval({x: [1, 2, 3, 4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(W.eval())
print(b.eval())
print(a.eval())


sess.close()

