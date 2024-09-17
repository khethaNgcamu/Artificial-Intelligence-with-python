import matplotlib.pyplot as plt
import numpy as np

list_1 = [0, 1, 2, 3, 4, 5]
list_2 = [1, 3, 5, 7, 9, 11]

x = np.array(list_1)
y = np.array(list_2)

plt.plot(x, y)

plt.show()
