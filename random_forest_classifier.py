import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#Load the dataset
iris_data = load_iris()
X = iris_data.data  #Features
y = iris_data.target    #Target

#Split the data set into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

#Initialise the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#Train the model
rf_classifier.fit(X_train, y_train)

#Make prediction
y_pred = rf_classifier.predict(X_test)

#Evaluate the model
accurancy = accuracy_score(y_test, y_pred)
print(f"random forest accurancy score is {accurancy * 100:.2f}%")

#Get feature importance
f_importances = rf_classifier.feature_importances_
feature_names = iris_data.feature_names

#Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_names, f_importances, color='green')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature importance in Random forest classifier")
plt.show()
