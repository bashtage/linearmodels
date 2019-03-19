from linearmodels.panel.lsmr.lsmr import experiment, LSMR
import numpy as np

y = np.random.randn(1000)
x = np.random.randn(1000, 5)
LSMR(y, x)

# experiment()
# experiment()

x = [[1.0, 0.0, -1.0],
     [0.0, 2.0, 0.0],
     [2.0, 0.0, 2.0],
     [5.0, 3.0, -2.0],
     [0.0, 0.0, 6.0]]
x = np.array(x)
y = np.ones(5)
l = LSMR(y, x)
print(l.beta)

from scipy.sparse.linalg import lsmr as splsmr

print(splsmr(x, y)[0])