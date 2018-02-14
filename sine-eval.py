import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.name_scope('placeholders')
x = tf.placeholder('float', [None, 1])
y = tf.placeholder('float', [None, 1])

tf.name_scope('neural_network')
x1 = tf.contrib.layers.fully_connected(x, 100)
x2 = tf.contrib.layers.fully_connected(x1, 100)
result = tf.contrib.layers.fully_connected(x2, 1, activation_fn=None)

loss = tf.nn.l2_loss(result - y)

tf.name_scope('optimizer')
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
#sess.run(tf.global_variables_initializer())
tf.train.Saver().restore(sess, "d:/dev/deeplearning/tensorflow-test/model.ckpt")

### Train the network
##for i in range(10000):
##    xpts = np.random.rand(100) * 10
##    ypts = np.sin(xpts)
##
##    _, loss_result = sess.run([train_op, loss],
##                                feed_dict={x: xpts[:, None],
##                                y: ypts[:, None]})
##
##    print('iteration {}, loss={}'.format(i, loss_result))
##save_path = tf.train.Saver().save(sess,
##                                  "d:/dev/deeplearning/tensorflow-test/model.ckpt")

xx = np.random.rand(1000)*10
yy = sess.run(result, feed_dict={x: xx[:,None]})
yyy = np.squeeze(yy)
plt.plot(xx, yyy, '.')
plt.show()