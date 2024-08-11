from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the housing Housing dataset
housing = fetch_california_housing()
housing_df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
housing_df['MEDV'] = housing.target

# Save the dataset to a CSV file
housing_df.to_csv('housing_dataset.csv', index=False)
