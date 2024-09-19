import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

#Get the file
file_path = 'student_risk_data.csv'
data_frame = pd.read_csv(file_path)

#Encoding categorical data using one hot encoder
categorical_data = ['gender','participation', 'parental_support', 'health_issues']
encoded_data = pd.get_dummies(data_frame[categorical_data], drop_first=True)

# Convert boolean to numeric (0 and 1)
encoded_data = encoded_data.astype(int)

#arranging numerical colomns and removing at_risk
numerical_data = ['age', 'attendance_rate', 'homework_completion_rate',
                   'test_scores', 'disciplinary_incidents', 'family_income',
                   'extracurriculars', 'mental_health_score']

#Combining data encoded and numerical data
data_frame_combined = pd.concat([data_frame[numerical_data], encoded_data], axis=1)

#Displaying few rows
print(data_frame_combined.head())

#handling missing values (if any) using simpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data_frame_combined), columns=data_frame_combined.columns)

#Scaling the data
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)

#Train the model

gmm_model = GaussianMixture(n_components=10, random_state=42)
gmm_model.fit(scaled_data)

#Add the GMM cluster labels to the DataFrame
data_frame_combined['gmm_cluster'] = gmm_model.predict(scaled_data)

# Display the cluster assignments
print(data_frame_combined[['gmm_cluster']].head())

#Visualising data using two features: attendance_rate and test_scores
plt.scatter(data_frame_combined['attendance_rate'],
            data_frame_combined['test_scores'],
            c=data_frame_combined['gmm_cluster'], cmap='plasma')

plt.xlabel('Attendance Rate')
plt.ylabel('Test Scores')
plt.title('GMM Clustering: Attendance Rate vs. Test Scores')
plt.show()

# Get the cluster probabilities
cluster_probs = gmm_model.predict_proba(scaled_data)

# Display the probabilities for the first few students
print("Cluster probabilities for each student:\n", cluster_probs[:5])

#adding at risk column on our data frame combined
data_frame_combined['at_risk'] = data_frame['at_risk']

# Compare the clusters with 'at_risk' status
risk_comparison = data_frame_combined.groupby('gmm_cluster')['at_risk'].mean()
print("\nGMM Cluster vs At-Risk:\n", risk_comparison)

# Convert the soft probabilities to hard labels by taking the cluster with the highest probability
hard_labels_from_probs = np.argmax(cluster_probs, axis=1)

# Calculate silhouette score to evaluate the clustering quality
silhouette_avg = silhouette_score(scaled_data, hard_labels_from_probs)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# # Apply DBSCAN clustering
# dbscan = DBSCAN(eps=1.0, min_samples=5)  # Adjust eps and min_samples for your data
# dbscan_labels = dbscan.fit_predict(scaled_data)

# # Check the unique labels
# print(set(dbscan_labels))  # Check if DBSCAN is detecting multiple clusters

# # Only calculate the Silhouette Score if DBSCAN finds more than 1 cluster
# if len(set(dbscan_labels)) > 1:
#     silhouette_avg = silhouette_score(scaled_data, dbscan_labels)
#     print(f"DBSCAN Silhouette Score: {silhouette_avg:.3f}")
# else:
#     print("DBSCAN did not form enough clusters for Silhouette Score calculation.")


# # Calculate Silhouette Score
# silhouette_avg = silhouette_score(scaled_data, dbscan_labels)
# print(f"DBSCAN Silhouette Score: {silhouette_avg:.3f}")
