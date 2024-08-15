import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#load the data
file_path = 'C:/Users/Khetha/Desktop/Neural-Network-Exercises/housing_dataset.csv'
data_frame = pd.read_csv(file_path)

#Handling missing data (if any)
data_frame = data_frame.fillna(data_frame.mean())

#print the first few rows
print("The first few rows of the data set: ")
print(data_frame.head())

#Splitting the data into features and target
X = data_frame.drop('MEDV', axis=1)
y = data_frame['MEDV']

#Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scalling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Initialise the model
lin_reg = LinearRegression()

#training the model
lin_reg.fit(X_train, y_train)

#Make prediction
y_pred = lin_reg.predict(X_test)

#Evaluate the model
print("================================================================")
print('Mean square error: ', mean_squared_error(y_test, y_pred))
print('R2 score: ', r2_score(y_test, y_pred))
