from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#load dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

#Splitting dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

#initialise decision tree clasifier
decision_tree = DecisionTreeClassifier(random_state=42)

#Train the model
decision_tree.fit(X_train, y_train)

#Make prediction
y_pred = decision_tree.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision tree accurancy score: {accuracy:.2f}")