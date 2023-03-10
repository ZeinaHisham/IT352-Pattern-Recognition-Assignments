import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

fig, ax = plt.subplots(1,3,figsize=(12,6))
colors = ["navy", "turquoise", "darkorange", "lightsteelblue", "indianred"]

# Load the Data
iris = load_iris()
x, y, target_names = iris.data, iris.target, iris.target_names
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.15, test_size=0.85, random_state=42)

# Plot Data
ax[1].set_title("IRIS Dataset")
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax[1].scatter(
        x[y == i, 0], x[y == i, 1], alpha=0.8, color=color, label=target_name
    )
ax[1].legend(loc='best')

# Implement Built-in LDA
model = LDA()
trx = model.fit_transform(train_x, train_y)
tstx = model.transform(test_x)
pred_y = model.predict(test_x)

# Some Outputs
print('\nUsing the LDA Built-in Function: \nAccuracy Score: ', model.score(test_x, test_y)*100 ,'%\nWrong Classifications:')
for i in range(len(pred_y)):
    if pred_y[i] != test_y[i]:
        print('Predicted Class: ', pred_y[i], ', Actual Class: ', test_y[i])
print('------------------------------\n')

# Plot Built-in LDA
ax[0].set_title("Built-in LDA")
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax[0].scatter(
        test_x[pred_y == i, 0], test_x[pred_y == i, 1], alpha=0.8, color=color, label=target_name
    )
ax[0].legend(loc="best")

# Implement LDA from Scratch
eta = 0.21 # Learning Step

def get_weights(class_no, threshold):
    w = []
    for i in range(5):  # Initialize Weights Randomly
        w.append(random.randint(-3, 3))

    for x in range(threshold):
        matches = 0
        for i in range(len(train_x)):
            dx = w[0] * train_x[i][0] + w[1] * train_x[i][1] + w[2] * train_x[i][2] + w[3] * train_x[i][3] + w[4]
            if ((dx > 0) and (train_y[i]==class_no)):
                matches += 1
            elif ((dx < 0) and (train_y[i]==class_no)):
                for j in range(4):
                    w[j] += eta * train_x[i][j]
                w[4] += eta
            elif ((dx > 0) and (train_y[i] != class_no)):
                for j in range(4):
                    w[j] -= eta * train_x[i][j]
                w[4] -= eta
    return w

def test_data(w1, w2, w3):
    matches = 0
    wrong_class = ''
    new_class = ''
    undetermined = ''
    plot_y = test_y

    for i in range(len(test_x)):
        d1 = w1[0] * test_x[i][0] + w1[1] * test_x[i][1] + w1[2] * test_x[i][2] + w1[3] * test_x[i][3] + w1[4]
        d2 = w2[0] * test_x[i][0] + w2[1] * test_x[i][1] + w2[2] * test_x[i][2] + w2[3] * test_x[i][3] + w2[4]
        d3 = w3[0] * test_x[i][0] + w3[1] * test_x[i][1] + w3[2] * test_x[i][2] + w3[3] * test_x[i][3] + w3[4]
        if(d1 > 0) and (d2 < 0) and (d3 < 0) and test_y[i] == 0:
            matches += 1
            plot_y[i] = 0
        elif(d1 > 0) and (d2 <0) and (d3 < 0) and test_y[i] != 0:
            wrong_class += 'Test class: '+ str(test_y[i]+1) + ' misclassified as class 1\n'
            plot_y[i] = 0
        elif (d1 < 0) and (d2 > 0) and (d3 < 0) and test_y[i] == 1:
            matches += 1
            plot_y[i] = 1
        elif (d1 < 0) and (d2 > 0) and (d3 < 0) and test_y[i] != 1:
            wrong_class += 'Test class: '+ str(test_y[i]+1)+ ' misclassified as class 2\n'
            plot_y[i] = 1
        elif (d1 < 0) and (d2 < 0) and (d3 > 0) and test_y[i] == 2:
            matches += 1
            plot_y[i] = 2
        elif (d1 < 0) and (d2 < 0) and (d3 > 0) and test_y[i] != 2:
            wrong_class += 'Test class: '+ str(test_y[i]+1)+ ' misclassified as class 3\n'
            plot_y[i] = 2
        elif d1 > 0 and ((d2 > 0 and d3 < 0) or (d2 < 0 and d3 > 0)):
            c = 2 if d2 > 0 else 3
            undetermined += 'Test class : '+ str(test_y[i]+1)+ ' classified as class 1, '+ str(c) + '\n'
            plot_y[i] = 3
        elif d2 > 0 and ((d1 > 0 and d3 < 0) or (d1 < 0 and d3 > 0)):
            c = 1 if d1 > 0 else 3
            undetermined += str('Test class : ' + str(test_y[i]+1) + ' classified as class 2, ' + str(c) + '\n')
            plot_y[i] = 3
        elif d3 > 0 and ((d2 > 0 and d1 < 0) or (d2 < 0 and d1 > 0)):
            c = 2 if d2 > 0 else 1
            undetermined += str('Test class : '+ str(test_y[i]+1) + ' classified as class 3, '+ str(c) + '\n')
            plot_y[i] = 3
        elif d1 < 0 and d2 < 0 and d3 < 0:
            new_class += str('Test class : '+ str(test_y[i]+1) + ' classified as new class\n')
            plot_y[i] = 4

    return matches, wrong_class, undetermined, new_class, plot_y


w1 = get_weights(0, 500)
w2 = get_weights(1, 500)
w3 = get_weights(2, 500)

matches, wrong_class, undetermined, new_class, plot_y = test_data(w1, w2, w3)

# Output Data
print('w1 = ', w1, '\nw2 = ', w2, '\nw3 = ',w3)
print('\nCorrect Matches: ', matches, ' Acurracy Score: ' , matches/len(test_x)*100,'%\nWrong Classifications: ')
print(wrong_class ,'\nUndetermined Samples:\n', undetermined, '\nNew Class:\n',new_class)

labels = target_names
labels= np.append(labels, 'undetermined')
labels= np.append(labels, 'new class')

ax[2].set_title("Manual LDA")
for color, i, label in zip(colors, [0,1,2,3,4], labels):
    ax[2].scatter(
        test_x[plot_y == i, 0], test_x[plot_y == i, 1], alpha=0.8, color=color, label=label
    )
ax[2].legend(loc="best", shadow=False, scatterpoints=1)

plt.show()