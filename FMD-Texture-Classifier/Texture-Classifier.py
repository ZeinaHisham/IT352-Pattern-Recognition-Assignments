import cv2
import math
import os, sys
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops

def create_dataset(img_folder):
    img_data_array = []
    class_name = []
    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir)):
            if any([file.endswith(x) for x in ['.jpeg', '.jpg']]):
                image_path = os.path.join(img_folder, dir, file)
                image = cv2.imread(image_path,0)
                img_data_array.append(image)
                class_name.append(dir)
    return img_data_array, class_name

def glcm(x, property):
    glcm_prop = []
    for i in range(len(x)):
        glcm = graycomatrix(x[i], distances=[2], angles=[90], normed=True)
        glcm_prop.append(graycoprops(glcm, property)[0, 0])
    return glcm_prop

def knn(xtr, ytr, xts, yts):
    matches = 0
    for i in range(len(xts)):
        dist = []
        for j in range(len(xtr)):
            dist.append(math.sqrt((xtr[j]-xts[i])**2))
        #print('Test Texture: ', yts[i], ', Training Texture: ', ytr[dist.index(min(dist))])
        if ytr[dist.index(min(dist))] == yts[i]:
            matches += 1
    return matches

#Create Dataset
X_Data, Y_Data = create_dataset("FMD/image")

#Split into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_Data, Y_Data, train_size=0.70, test_size=0.30, random_state=42)

#Define properties
props = [
    'contrast',
    'homogeneity',
    'energy',
    'correlation'
]

#Create GLCM and extract properties
for i in props:
    glcm_prop_train = glcm(X_train, i)
    glcm_prop_test = glcm(X_test, i)
    accuracy = knn(glcm_prop_train, Y_train, glcm_prop_test, Y_test)
    print('Accuracy score using ' + i + ': ' + str(accuracy / (len(Y_test)) * 100) + '%')
