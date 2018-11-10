#importing the different packages 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt, matplotlib.image as mpimg

#splitting train.csv into train and test set
train_csv = pd.read_csv("C:/Users/shrav/Desktop/things to finish/KDD/Final_Project-Image_Recognition/all/train.csv")
features = train_csv.columns[1:]
X = train_csv[features]
Y = train_csv['label']
X_train, X_test, Y_train, Y_test = train_test_split(X/255., Y, test_size=0.25, random_state=0)


#Random Forest classifer on train set
classifier_rf = RandomForestClassifier()
classifier_rf.fit(X_train, Y_train)

#Prediction and accuracy
Y_pred_rf = classifier_rf.predict(X_test)
accuracy_rf = accuracy_score(Y_test, Y_pred_rf)

print(accuracy_rf)
