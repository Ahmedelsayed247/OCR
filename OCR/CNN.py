from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#cutting image into images
def imaged_grid(img , row , col ):

    x , y= img.shape 
    assert x % row == 0, x % row .format(x, row)
    assert y % col == 0, y % col.format(y, col)
    
    
    return (img.reshape ( x //row, row, -1, col)
               .swapaxes(1,2)
               .reshape(-1, row, col))

def get_centroid(img):
 
    feature_vector = []
 
    for grid in imaged_grid(img , 4 , 4 ) :
        
        X_center = 0 
        Y_center = 0 
        sum = 0

        for x in range(4):
          for y in range(4):
            X_center=X_center+x*grid[x][y]
            Y_center=Y_center+y*grid[x][y]
            sum+=grid[x][y]

        if sum == 0 :
            feature_vector.append(0)
            feature_vector.append(0)
        
        else :
          feature_vector.append( X_center/ sum )
          feature_vector.append(Y_center/ sum )
     
    return np.array(feature_vector)

#classify featuers by KNN from scratch
class KNN:
    distances = [[]*2]
    final_label = []
    def getDistance(self,test_vector,train_feature_vectors,train_labels): 
        for i in range(len(train_feature_vectors)):
            distance = np.linalg.norm(test_vector-train_feature_vectors[i]) #calculate the distance between test vector and train vectors
            self.distances.append([distance,train_labels[i]]) #append the distance and label to the distances array
        self.distances = sorted(self.distances, key=lambda x:x[0]) #sort the distances array by the distance
        return self.distances
    def getLabel(self,k):
        labels = []
        for i in range(k):
            labels.append(self.distances[i][1]) 
        return labels
    def getNearestNeighbor(self,k):
        labels = self.getLabel(k)
        return max(set(labels), key=labels.count)
    def Classifier(self,k,train_features, test_features, Trainlabels):
        for i in range(len(test_features)):
            self.distances = []
            self.getDistance(test_features[i],train_features,Trainlabels) #fill distances list with distances and labels 
            self.final_label.append(self.getNearestNeighbor(k)) #get the label of the nearest k neighbor
        return self.final_label
    
    
def Knn(train_features, test_features, Trainlabels):# KNN with Smile Detection (Built-in)
    knn = KNeighborsClassifier(50, metric='euclidean')
    
    #fitting data
    knn.fit(train_features, Trainlabels) 
    prediction = knn.predict(test_features)  
    return prediction
    
def load_data(path):
    data = []
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        data.append(img_array)
        if(len(data)>=100):
            break
    return data

def getAccuracy(prediction, test_labels):
    wrong_classifier = 0
    for i in range(len(test_labels)):
        if prediction[i] != test_labels[i]:
           wrong_classifier+=1
    return (100-(wrong_classifier/len(test_labels))*100)
    
#__________________Main_______________

#load data from negative and positive folder
positiveData = load_data("positives_Smile")
negativeData = load_data("negatives_No_Smile")

X = np.concatenate((positiveData, negativeData), axis=0)
y = np.concatenate((np.ones(len(positiveData)), np.zeros(len(negativeData)))) #positive = 1, negative = 0

#split data into training and testing
Datatrain, Datateast, train_labels, Teastlabels = train_test_split(X, y, test_size=0.2)

#shape of dataset
self = KNN()
train_features = [get_centroid(img)  for img in Datatrain  ]
test_features = [get_centroid(img)  for img in Datateast ]

KNN_Prediction=KNN.Classifier(self,9,train_features, test_features, train_labels) #KNN From Scratch with 4 nearest neighbors
print("Scratch Accuracy = " , getAccuracy(KNN_Prediction, Teastlabels),"%") #get accuracy of KNN From Scratch with 4 nearest neighbors
 
Knn_prediction_BuiltIn = Knn(train_features, test_features , train_labels )# KNN with Smile Detection (Built-in)
print("Accuracy = ",  getAccuracy(Knn_prediction_BuiltIn, Teastlabels),"%")

    

