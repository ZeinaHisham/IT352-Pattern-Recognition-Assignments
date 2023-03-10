import math
import keras
import tensorflow
import numpy as np
from keras.datasets import mnist
from skimage.util.shape import view_as_blocks

# Calculate Centroid
def centroid(block):
    x = 0
    y = 0
    xy = 0
    for i in range(7):
        for j in range(7):
            x += i * block[i][j]
            y += j * block[i][j]
            xy += block[i][j]
    x = x / xy if xy > 0 else 0
    y = y / xy if xy > 0 else 0
    return x, y

# Calculate Euclidean Distance
def distance(tr_fv, tst_fv):
    d = 0
    for i in range(len(tr_fv)):
        for j in range(len(tr_fv[i])):
            d += (tst_fv[i][j] - tr_fv[i][j]) ** 2
    return math.sqrt(d)

def extractfv(x):
    x_blocks = view_as_blocks(x[i], block_shape=(7, 7))
    fv_image = np.empty()
    for j in range(4):
        for k in range(4):
            fv_image.append(centroid(x_blocks[j][k]))
    return fv_image

# Load the Dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Take 1000 samples for testing
xts = test_x[:1000, :, :]
yts = test_y[:1000]

# Filter for 0-9 and concatenate the first 1000 samples of each digit
for i in range(10):
    X_train, Y_train = train_x[np.where(train_y == i)], train_y[train_y == i]
    X_train = X_train[:1000, :, :]
    Y_train = Y_train[:1000]
    if i == 0:
        xtr = X_train
        ytr = Y_train
    else:
        xtr = np.concatenate((xtr, X_train))
        ytr = np.concatenate((ytr, Y_train))

#Print - Should output 10000 training samples and 1000 testing samples
print('X_train: ' + str(xtr.shape))
print('Y_train: ' + str(ytr.shape))
print('X_test:  '  + str(xts.shape))
print('Y_test:  '  + str(yts.shape))

# Create feature vector for the training set
train_fv = []
for i in range(len(xtr)):
    train_fv.append(extractfv(xtr))

# Create feature vector for the testing set
test_fv = []
for i in range(len(xts)):
    test_fv.append(extractfv(xts))

matches = 0

# Find Matching Image
for i in range(len(test_fv)):
    dist = []
    for j in range(len(train_fv)):
        dist.append(distance(train_fv[j], test_fv[i]))
    print('Test Number: ', yts[i], ', Training Number: ', ytr[dist.index(min(dist))])
    if ytr[dist.index(min(dist))] == yts[i]:
        matches += 1
#print(matches)
print('Accuracy Score: ' + str(matches / (len(xts)) * 100) + '%')