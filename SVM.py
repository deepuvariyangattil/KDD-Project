#importing all packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt, matplotlib.image as mpimg


#splitting train.csv file to train and test sets
train_csv = pd.read_csv("C:/Users/shrav/Desktop/things to finish/KDD/Final_Project-Image_Recognition/all/train.csv")
features = train_csv.columns[1:]
X = train_csv[features]
Y = train_csv['label']
X_train, X_test, Y_train, Y_test = train_test_split(X/255., Y, test_size=0.25, random_state=0)

#SVM classifier

classifier_svm = LinearSVC()
classifier_svm.fit(X_train, Y_train)

#prediction and accuracy
Y_pred_svm = classfier_svm.predict(X_test)
accuracy_svm = accuracy_score(Y_test, Y_pred_svm)
print(accuracy_svm) 


