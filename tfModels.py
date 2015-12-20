import numpy as np
import tensorflow as tf

"""
code for running tests of each model

import numpy as np
X = np.random.randn(10000, 5)
Y = np.random.randn(10000, 1) + 0.5 * X[:, 0:1] - 0.25 * X[:, 1:2]
m = linReg(X[0::2, :], Y[0::2], X[1::2, :], Y[1::2])

"""


def linReg(Xtrain, Ytrain, Xtest, Ytest):
    """
    Linear regression built in tensor flow
    """

    numIter = 5000
    dim = Xtrain.shape[1]
    totalTrainVar = np.mean(Ytrain**2)
    totalTestVar = np.mean(Ytest**2)
    with tf.Session() as sess:
        x = tf.placeholder("float", shape=[None, dim])
        y_ = tf.placeholder("float", shape=[None, 1])

        W = tf.Variable(tf.random_uniform([dim, 1]))
        # W = tf.Print(W, [W], 'W:')

        b = tf.Variable(tf.random_uniform([1, 1]))
        # b = tf.Print(b, [b], 'b:')

        y = tf.matmul(x, W) + b
        # y = tf.Print(y, [y], 'y:')

        mse = tf.reduce_mean(tf.square(y_ - y))
        sess.run(tf.initialize_all_variables())

        # train_step = tf.train.MomentumOptimizer(1.0e-2, 0.1).minimize(mse)
        train_step = tf.train.AdamOptimizer().minimize(mse)
        sess.run(tf.initialize_all_variables())

        for i in range(numIter):
            train_step.run(feed_dict={x: Xtrain, y_: Ytrain}, session=sess)
            if (i % 500) == 0:
                trainError = mse.eval(feed_dict={x: Xtrain, y_: Ytrain}, session=sess)
                trainR2 = (totalTrainVar - trainError) / totalTrainVar
                print("Train r2: {:.2f}".format(trainR2))
        testError = mse.eval(feed_dict={x: Xtest, y_: Ytest}, session=sess)
        testR2 = (totalTestVar - testError) / totalTestVar
        print("Test r2: {:.2f}".format(testR2))

        statModel = [W.eval(session=sess), b.eval(session=sess), testR2]
    return statModel



if False:
    def convNet(Xtrain, Ytrain, Xtest, Ytest):
        """
        Convolutional netowrk built in tensor flow
        """

        numIter = 5000
        dim = Xtrain.shape[1]
        totalTrainVar = np.mean(Ytrain**2)
        totalTestVar = np.mean(Ytest**2)
        with tf.Session() as sess:
            x = tf.placeholder("float", shape=[None, dim])
            y_ = tf.placeholder("float", shape=[None, 1])

            W = tf.Variable(tf.random_uniform([dim, 1]))
            # W = tf.Print(W, [W], 'W:')

            b = tf.Variable(tf.random_uniform([1, 1]))
            # b = tf.Print(b, [b], 'b:')

            y = tf.matmul(x, W) + b
            # y = tf.Print(y, [y], 'y:')

            mse = tf.reduce_mean(tf.square(y_ - y))
            sess.run(tf.initialize_all_variables())

            # train_step = tf.train.MomentumOptimizer(1.0e-2, 0.1).minimize(mse)
            train_step = tf.train.AdamOptimizer().minimize(mse)
            sess.run(tf.initialize_all_variables())

            for i in range(numIter):
                train_step.run(feed_dict={x: Xtrain, y_: Ytrain}, session=sess)
                if (i % 500) == 0:
                    trainError = mse.eval(feed_dict={x: Xtrain, y_: Ytrain}, session=sess)
                    trainR2 = (totalTrainVar - trainError) / totalTrainVar
                    print("Train r2: {:.2f}".format(trainR2))
            testError = mse.eval(feed_dict={x: Xtest, y_: Ytest}, session=sess)
            testR2 = (totalTestVar - testError) / totalTestVar
            print("Test r2: {:.2f}".format(testR2))

            statModel = [W.eval(session=sess), b.eval(session=sess), testR2]
        return statModel



    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    # First layer

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    # Second layer

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # Dense layer

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # Dropout

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # Readout layer

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # Train/evaluate

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step {:.2f}, training accuracy {}".format(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy {:.2f}".format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
