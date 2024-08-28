import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Step 2: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],        # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],        # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],          # Minimum samples required at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'] # Number of features to consider for the best split
}

# Step 4: Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Step 5: Initialize GridSearchCV to optimize the hyperparameters
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Step 6: Fit the model on the training data
grid_search_rf.fit(X_train, y_train)

# Step 7: Print the best parameters found by GridSearchCV
print("Best parameters found: ", grid_search_rf.best_params_)

# Step 8: Use the best estimator to make predictions on the test data
best_rf_classifier = grid_search_rf.best_estimator_
y_pred_optimized_rf = best_rf_classifier.predict(X_test)

# Step 9: Evaluate the optimized Random Forest model
optimized_rf_accuracy = accuracy_score(y_test, y_pred_optimized_rf)
print(f"Optimized Random Forest Accuracy: {optimized_rf_accuracy * 100:.2f}%")

# Step 10: Feature Importance Plot
importances = best_rf_classifier.feature_importances_
feature_names = iris.feature_names

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances, color='blue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Optimized Random Forest Classifier')
plt.show()
