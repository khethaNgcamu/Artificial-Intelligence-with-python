from sklearn.datasets import load_iris
import pandas as pd

#Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

#save a dataset to a csv file
iris_df.to_csv('iris_dataset.csv', index=False)
