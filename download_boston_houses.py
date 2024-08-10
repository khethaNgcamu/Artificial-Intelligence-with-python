from sklearn.datasets import load_boston
import pandas as pd

# Load the Boston Housing dataset
boston = load_boston()
boston_df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
boston_df['MEDV'] = boston.target

# Save the dataset to a CSV file
boston_df.to_csv('boston_housing_dataset.csv', index=False)
