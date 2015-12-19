import os as os
import glob as glob

import skimage.data as skimData
from skimage.color import rgb2grey
from skimage.transform import resize

import tensorflow as tf

import matplotlib.pyplot as plt

rootDir = '/Users/mcmenamin/GitHub/tensorFlowFaces/'

######################################
#
# Reading in all face images, resizing
#

trainFnames = glob.glob(rootDir + 'SaidAndTodorov_Model/faces_training_jpg/[m,f]*.jpg')
testFnames = glob.glob(rootDir + 'SaidAndTodorov_Model/faces_testing_jpg/[m,f]*.jpg')


def imageProcess(x):
    return resize(rgb2grey(skimData.load(x)), [128, 128])

train_images = [imageProcess(f) for f in trainFnames]
test_images = [imageProcess(f) for f in testFnames]

train_images = [t.ravel() for t in train_images]
test_images = [t.ravel() for t in test_images]

train_images = [t / np.sqrt(np.mean(t**2)) for t in train_images]
test_images = [t / np.sqrt(np.mean(t**2)) for t in test_images]


######################################
#
# Reading in attractiveness ratings for each face
#

attr_maleFaces_train = np.loadtxt(rootDir + 'SaidAndTodorov_Model/FrM_attractivenessratings_formatlab.csv', delimiter=',')[:, 0]
attr_femlFaces_train = np.loadtxt(rootDir + 'SaidAndTodorov_Model/MrF_attractivenessratings_formatlab.csv', delimiter=',')[:, 0]
attr_maleFaces_test = np.loadtxt(rootDir + 'SaidAndTodorov_Model/validationresultsFrM_formatlab.csv', delimiter=',')[1:, 0]
attr_femlFaces_test = np.loadtxt(rootDir + 'SaidAndTodorov_Model/validationresultsMrF_formatlab.csv', delimiter=',')[1:, 0]


train_attr = []
for f in trainFnames:
    f = f.split('/')[-1]
    num = int(f[1:-4])
    if f[0] == 'm':
        train_attr.append(attr_maleFaces_train[num])
    elif f[0] == 'f':
        train_attr.append(attr_femlFaces_train[num])
    else:
        print('bad gender?')

test_attr = []
for f in testFnames:
    f = f.split('/')[-1]
    num = int(f[1:-4])
    if f[0] == 'm':
        test_attr.append(attr_maleFaces_test[num])
    elif f[0] == 'f':
        test_attr.append(attr_femlFaces_test[num])
    else:
        print('bad gender?')




####################################
#
# Linear regression
#

sess = tf.Session()

x = tf.placeholder("float", shape=[None, 128**2])
y_ = tf.placeholder("float", shape=[None, 1])

W = tf.Variable(tf.random_uniform([128**2, 1]))
b = tf.Variable(tf.random_uniform([1, 1]))

y = tf.add(tf.matmul(x, W), b)
rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))
# rmse = (tf.pow(y_ - y, 2))

sess.run(tf.initialize_all_variables())

%cpaste

batchSize = 4000
stepSize = len(train_attr) // batchSize

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(rmse)
for i in range(2000):
    batchIdx = np.random.random_integers(0, len(train_attr) - 1, batchSize)
    tmpX = np.vstack([train_images[i] for i in batchIdx])
    tmpY = np.vstack([train_attr[i] for i in batchIdx])
    train_step.run(feed_dict={x: tmpX, y_: tmpY}, session=sess)
    if (i % 100) == 0:
        trainError = rmse.eval(feed_dict={x: tmpX, y_: tmpY}, session=sess)
        # trainError = np.sqrt(np.mean(trainError))
        print("Performance on train step: {:.2f}".format(trainError))
testError = rmse.eval(feed_dict={x: np.vstack(test_images), y_: np.vstack(test_attr)}, session=sess)
# testError = np.sqrt(np.mean(testError))
print("Performance on test set: {:.2f}".format(testError))

--



ypred = y.eval(feed_dict={x: np.vstack(train_images)}, session=sess)
plt.scatter(train_attr, ypred)
from scipy.stats import pearsonr
pearsonr(np.vstack(train_attr).ravel(), ypred.ravel())

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(tmpX, tmpY)
print(model.score(tmpX, tmpY))
print(model.score(np.vstack(test_images), np.vstack(test_attr)))


"""
The result is pretty crappy. That means that we're not just comparing each face to
a single 'attractive' template in a pixel-wise manner. Let step up the model complexity
and do a convolutional network!
"""




####################################
#
# Multilayer convolutional network
#



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
