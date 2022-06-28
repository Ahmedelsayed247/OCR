from keras.datasets import mnist
import tensorflow as tf
from matplotlib import pyplot
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris



def featureVector(train_image):
    ret, thresh1 = cv2.threshold(train_image, 40, 255, cv2.THRESH_BINARY)
    for i, row in enumerate(thresh1):
        for j, value in enumerate(row):
            if value == 255:
                start_point = (i, j)
                break
        else:
            continue
        break

    directions = [0, 1, 2,
                  7, 3,
                  6, 5, 4]



    dir2idx = dict(zip(directions, range(len(directions))))

    change_i = [-1, 0, 1,  # x or columns
                -1, 1,
                -1, 0, 1]

    change_j = [-1, -1, -1,  # y or rows
                0, 0,
                1, 1, 1]

    border = []
    chain = []
    curr_point = start_point
    for direction in directions:
        idx = dir2idx[direction]
        if start_point[0] + change_i[idx] < 28 and start_point[1] + change_j[idx] < 28:
            new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])
            if thresh1[new_point] != 0:  # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        else:
            new_point = (start_point[0] + change_i[idx] - 1, start_point[1] + change_j[idx] - 1)
            # print("elseeslelelelelelelelelleleleleellele")
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point

    count = 0
    while curr_point != start_point:
        # figure direction to start search
        b_direction = (direction + 7) % 8
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []

        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])

            if new_point[0] < 27 and new_point[1] < 27:
                if thresh1[new_point] != 0:  # if is ROIv
                    # if(len(chain)<3):
                    chain.append(direction)
                    border.append(new_point)

                    curr_point = new_point
                    break
        if count == 100: break
        count += 1



    return chain, border



# read mnist data
(training_X, training_Y), (test_X, test_Y) = mnist.load_data()

training_images = training_X
training_labels = training_Y
test_images = test_X
test_labels = test_Y


# split && centroid training data
training_featureVectors = []
for i in range(len(training_X)):

    chain,border = featureVector(training_images[i])
    tmp1=np.array(chain)
    tmp1.resize(100)
    # print(tmp1)
    training_featureVectors.append(tmp1)


# split && centroid test data
test_featureVectors = []
for i in range(len(test_X)):
    chain,border = featureVector(test_images[i])
    tmp1 = np.array(chain)
    tmp1.resize(100)
    # print(tmp1)
    test_featureVectors.append(tmp1)
    ret, thresh1 = cv2.threshold(test_images[i], 40, 255, cv2.THRESH_BINARY)
    # pyplot.imshow(thresh1, cmap=pyplot.get_cmap('gray'))
    # pyplot.plot([k[1] for k in border], [k[0] for k in border])
    # pyplot.show()


error=[]

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(training_featureVectors, training_labels)

# Calculate the accuracy of the model
# print(knn.predict(test_featureVectors))
score= knn.score(test_featureVectors, test_labels)
accuracy= round(score*100)
print('Test accuracy:', accuracy, '%')

# classify training_featureVectors, training_labels with artificial neural network

sc = StandardScaler()
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(400, activation= 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
training_featureVectors = sc.fit_transform(training_featureVectors)
test_featureVectors = sc.transform(test_featureVectors)
r = model.fit(training_featureVectors, training_labels, validation_data=(test_featureVectors, test_labels), epochs=20)

