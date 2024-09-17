import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#Load the file
file_path = 'student_risk_data.csv'
data_frame = pd.read_csv(file_path)

#Dislaying the few rows
#print(data_frame.head)

#Encoding the categorical data using One-Hot Encoding
categorical_columns = ['gender', 'participation', 'parental_support', 'health_issues']
data_encoded = pd.get_dummies(data_frame[categorical_columns], drop_first=True)

# Convert boolean to numeric (0 and 1)
data_encoded = data_encoded.astype(int)

# Extracting numerical columns (except 'at_risk')
numeric_columns = ['age', 'attendance_rate', 'homework_completion_rate',
                   'test_scores', 'disciplinary_incidents', 'family_income',
                   'extracurriculars', 'mental_health_score']

# Create a new dataframe combining both numeric and one-hot encoded categorical columns
data_frame_combined = pd.concat([data_frame[numeric_columns], data_encoded], axis=1)

#Displaying few rows
print(data_frame_combined.head())

#Handling missing data using SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data_frame_combined), columns=data_frame_combined.columns)

# Display the data after handling missing values
print("\nData after handling missing values:\n", data_imputed.head())

# #scale numeric data
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)

# # Add the target 'at_risk' column back to the dataset
# data_final = pd.concat([data_scaled, data_frame['at_risk']], axis=1)

# # Display the final preprocessed dataset
print("\nFinal Preprocessed Data:\n", data_scaled.head())

#Create the model
k_means_model = KMeans(n_clusters=3, random_state=42)
k_means_model.fit(data_scaled)

#Add cluster labels to the orinal data
data_frame_combined['cluster'] = k_means_model.labels_

# Display the cluster assignments
print("\nCluster assignments:\n", data_frame_combined[['cluster']].head())

# Get the mean feature values for each cluster
cluster_means = data_frame_combined.groupby('cluster').mean()
print("\nCluster average:\n",cluster_means)

#Addading "at_risk" to data_frame_combined
data_frame_combined['at_risk'] = data_frame['at_risk']

# Compare clusters with at_risk label
risk_comparison = data_frame_combined.groupby('cluster')['at_risk'].mean() * 100
#print("\n the average at-risk score for students in each cluster:\n",risk_comparison)
print("\nthe average at-risk score for students in each cluste:")
for cluster, risk in risk_comparison.items():
    print(f"Cluster {cluster}: {risk:.2f}% of students are at risk.")

# Visualize the clusters (using 2 features for visualization)
# Using 'attendance_rate' and 'test_scores' for visualization
plt.scatter(data_frame_combined['attendance_rate'], data_frame_combined['test_scores'],
            c=data_frame_combined['cluster'], cmap='viridis')

# Calculate silhouette score to evaluate the clustering quality
silhouette_avg = silhouette_score(data_scaled, k_means_model.labels_)
print(f"Silhouette Score: {silhouette_avg:.3f}")

plt.xlabel('Attendance Rate')
plt.ylabel('Test Scores')
plt.title('K-Means Clustering: Attendance Rate vs. Test Scores')
plt.show()
