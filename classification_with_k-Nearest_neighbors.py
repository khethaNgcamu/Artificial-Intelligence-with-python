import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Load dataset
file_path = 'C:/Users/Khetha/Desktop/Neural-Network-Exercises/iris_dataset.csv'
data_frame = pd.read_csv(file_path)

#Display the first few rows of the data set
print("Firfst few rows of the data set: ")
print(data_frame.head())

# Handling missing value (if any)
data_frame = data_frame.fillna(data_frame.mean())

#Splitting data into features and target
X = data_frame.drop('species', axis=1)
y = data_frame['species']

#Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature scalling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#initiate the model
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model
knn.fit(X_train, y_train)

#Make predictions
y_pred = knn.predict(X_test)

#Evaluate the model
print("Confusion matrix:\n ", confusion_matrix(y_test, y_pred))
print("Classification report:\n ", classification_report(y_test, y_pred))
print("Accuracy score: ", accuracy_score(y_test, y_pred))
