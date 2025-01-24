import numpy as np

import matplotlib.pyplot as plt

model_order = np.array([5,20,50,100]).astype(np.float64)

time_s = np.array([1.395439863204956,17.847306489944458,123.92729306221008,624.4677813053131])

plt.plot(model_order,time_s)
plt.show()