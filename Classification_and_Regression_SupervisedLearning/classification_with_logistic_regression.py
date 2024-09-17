import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.model_selection import cross_val_score


# Load dataset
file_path = 'C:/Users/Khetha/Desktop/Neural-Network-Exercises/iris_dataset.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

#handling missing values(if any)
df = df.fillna(df.mean())

#Encoding categorical variables (if any)
# labelEncoder = LabelEncoder()
# df['category_column'] = labelEncoder.fit_transform(df['category_column'])

#Splitting data into features and tagert
X = df.drop('species', axis=1)
y = df['species']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print the first few rows to check
# print("X_train:\n", X_train[:5])
# print("y_train:\n", y_train[:5])

#Initialiased the model
logistic_r_model = LogisticRegression(random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(logistic_r_model, X_train, y_train, cv=5, scoring='accuracy')
cv_accuracy = cv_scores.mean()
cv_std = cv_scores.std()
print(f"Cross-Validation Accuracy: {cv_accuracy}")
print(f"Cross-Validation Standard Deviation: {cv_std}")

#Train a Logistic Regression model
logistic_r_model.fit(X_train, y_train)

#make prediction
y_pred = logistic_r_model.predict(X_test)

# Example predictions
print("Predicted labels for the test set:")
print(y_pred)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
