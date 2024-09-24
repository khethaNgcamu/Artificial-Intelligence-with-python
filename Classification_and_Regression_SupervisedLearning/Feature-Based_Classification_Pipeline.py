import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#load the data
# data = pd.read_csv('feature_data.csv')
# X = data.drop('Label', axis=1)  # Features (drop the label column)
# y = data['Label']  # Labels

# Generate structured data where labels depend on the sum of key features
np.random.seed(42)
X = np.random.rand(150, 25)
y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)

#spliting data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the data
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)),  # Step 2: Select top k features (k=10)
    ('classifier', SVC(random_state=42))    #Support Vector Machines
   # ('classifier', RandomForestClassifier(random_state=42))  # Step 3: RandomForest model for classification
    # ('classifier', LogisticRegression(random_state=42))  # Logistic Regression model
])

#fit the pipeline on the training data
pipeline.fit(X_train, y_train)

#make a predict on a test set
y_pred = pipeline.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'the model accuracy is {accuracy * 100:.2f}%')

# Perform cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"Cross-validated accuracy: {scores.mean() * 100:.2f}%")
