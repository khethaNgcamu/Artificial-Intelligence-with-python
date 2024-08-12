from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Load dataset
file_path = 'C:/Users/Khetha/Desktop/Neural-Network-Exercises/iris_dataset.csv'
df = pd.read_csv(file_path)

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
print("X_train:\n", X_train[:5])
print("y_train:\n", y_train[:5])
