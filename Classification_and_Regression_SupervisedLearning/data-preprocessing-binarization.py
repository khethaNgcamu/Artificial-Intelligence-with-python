import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
                        [-1.2, 7.8, -6.1],
                        [3.9, 0.4, 2.1],
                        [7.3, -9.9, -4.5]])

#binarised data
data_binarised = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\n binarized data: \n", data_binarised)
