import numpy as np
import pandas as pd

# Generate 150 data points with 25-dimensional feature vectors
np.random.seed(42)  # For reproducibility
data = np.random.rand(150, 25)  # 150 rows, 25 columns

# Generate random binary labels (0 or 1) for classification
labels = np.random.randint(0, 2, 150)

# Convert to a DataFrame
df = pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(1, 26)])
df['Label'] = labels  # Add labels as a new column

# Save the data to a CSV file
df.to_csv('feature_data.csv', index=False)
print("Data has been saved to 'feature_data.csv'")
