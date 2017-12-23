import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

'''
one_hot means we are looking for something like:
    ans = [0,0,0,0,1,0] etc, where only ONE node is 'hot' (1) and the rest are
    all zero
'''

<<<<<<< HEAD

=======
>>>>>>> 23790bcb0c0309a745c7226a4b6baf78ab51fe48
n_nondes_hl1 = 500
n_nondes_hl2 = 500
n_nondes_hl3 = 500

n_classes = 10
# numbers 0-9
batch_size = 100
# we go through batches of 100 features at a time, so that we do not have to
# load everything into memory at the same time?

# Matrix is height x width
x = tf.placeholder('float', [None,784])
# imagine a 1 x 784 sized matrix, pixels are all moved into one row from square
# image
y = tf.placeholder('float')
# y is the label of x?

def neural_network_model(data):

    # using a bias because when an input is zero, then neurons do not fire,
    # which is not ideal. this ensures that some neurons will fire even when
    # input is all zeros.

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nondes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nondes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nondes_hl1, n_nondes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nondes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nondes_hl2, n_nondes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nondes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nondes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 - tf.nn.relu(l1)
    # relu is our treshold function (rectified linear is the real name)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 - tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 - tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # learning rate = 0.001, default
    optimizer = tf.train.AdamOptimizer().minimize(cost)
<<<<<<< HEAD
    # optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
=======

>>>>>>> 23790bcb0c0309a745c7226a4b6baf78ab51fe48
    hm_epochs = 20
    # how many epochs?

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # initializes all of the variables, the session is now live!

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # chunks through dataset for you... but in the real world we would
                # have to do this on our own. mnist does this automatically
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c
<<<<<<< HEAD
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
=======
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
>>>>>>> 23790bcb0c0309a745c7226a4b6baf78ab51fe48

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        # checks to see whether we are correct or not

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # casts correct to type float

        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
