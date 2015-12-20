import os as os
import glob as glob

import skimage.data as skimData
from skimage.color import rgb2grey
from skimage.transform import resize

import tensorflow as tf

import matplotlib.pyplot as plt

rootDir = '/Users/mcmenamin/GitHub/tensorFlowFaces/'

os.chdir(rootDir)
import tfModels as tfModels

######################################
#
# Reading in all face images, resizing
#


def imageProcess(x):
    return resize(rgb2grey(skimData.load(x)), [150, 150])

trainFnames = glob.glob(rootDir + 'SaidAndTodorov_Model/faces_training_jpg/[m,f]*.jpg')
testFnames = glob.glob(rootDir + 'SaidAndTodorov_Model/faces_testing_jpg/[m,f]*.jpg')

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

# import importlib
# importlib.reload(tfModels)

Xtrain = np.vstack(train_images)
Ytrain = np.vstack(train_attr)
Xtest = np.vstack(test_images)
Ytest = np.vstack(test_attr)

"""
# Use this code to drop out 'empty' pixels

goodPix = np.mean(Xtrain**2, axis=0) > 0.01
Xtrain = Xtrain[:, goodPix]
Xtest = Xtest[:, goodPix]
"""

# Simplify input using SVD to project into a lower-dimensional space

U, S, Vt = np.linalg.svd(Xtrain, full_matrices=False)

toKeep = S > 1
print('Keeping {} features ({:.1f}%)'.format(np.sum(toKeep), 100 * np.mean(toKeep)))

Xtrain_lodim = Xtrain.dot(Vt[toKeep, :].T) / S[toKeep].reshape(1, -1)
Xtest_lodim = Xtest.dot(Vt[toKeep, :].T) / S[toKeep].reshape(1, -1)

# Use linear regression to find 'attractive' face dimensions

linearModel = tfModels.linReg(Xtrain_lodim, Ytrain,
                              Xtest_lodim, Ytest)


"""
The result is pretty good if the faces are really downsampled (i.e., LSF features
can predict attractiveness with r2 ~ 0.45).

Let's make things more fun (and predictive) by getting a full convolutional network model!
"""




####################################
#
# Multilayer convolutional network
#

import importlib
importlib.reload(tfModels)


Xtrain = np.vstack(train_images)
Ytrain = np.vstack(train_attr)
Xtest = np.vstack(test_images)
Ytest = np.vstack(test_attr)

deepModel = tfModels.convNet(Xtrain, Ytrain,
                             Xtest, Ytest)
