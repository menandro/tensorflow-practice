import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder('float', [None, 1])
y = tf.placeholder('float', [None, 1])

x1 = tf.contrib.layers.fully_connected(x, 100)
x2 = tf.contrib.layers.fully_connected(x1, 100)
result = tf.contrib.layers.fully_connected(x2, 1, activation_fn=None)

loss = tf.nn.l2_loss(result - y)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
#restore model if exist
model_path = "d:/dev/deeplearning/tensorflow-test/model.ckpt"
checkpoint_path = "d:/dev/deeplearning/tensorflow-test/checkpoint.ckpt"
if tf.gfile.Exists(model_path + ".meta"):
    tf.train.Saver().restore(sess, model_path)
else:
    sess.run(tf.global_variables_initializer())

# Train the network
loss_plot = []
for i in range(10000):
    xpts = np.random.rand(100) * 10
    ypts = np.sin(xpts)

    _, loss_result = sess.run([train_op, loss],
                                feed_dict={x: xpts[:, None],
                                y: ypts[:, None]})

    print('iteration {}, loss={}'.format(i, loss_result))
    loss_plot.append(loss_result)
    
save_path = tf.train.Saver().save(sess, model_path)

#plot loss and evaluate
plt.plot(loss_plot, '.')
xx = np.random.rand(100)*10
yy = sess.run(result, feed_dict={x: xx[:,None]})
yyy = np.squeeze(yy)
plt.figure()
plt.plot(xx, yyy, '.')
plt.show()
