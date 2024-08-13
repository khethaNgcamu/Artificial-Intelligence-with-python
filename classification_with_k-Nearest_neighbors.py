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
X = data_frame.drop('species' axis=1)
y = data_frame['species']

#Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature scalling
scaler = StandardScaler()
X_train = scaler.fit_tramsforn(X_train)
X_test = scaler.transform(X_test)
