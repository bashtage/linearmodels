import numpy as np
import pandas as pd
from numpy.linalg import pinv

c1 = pd.Categorical(['a','a','a','b','b'])
c2 = pd.Categorical(['a','b','a','b','a'])
d1 = pd.get_dummies(c1,drop_first=False).astype(np.float64)
d2 = pd.get_dummies(c2,drop_first=False).astype(np.float64)
d1 = d1.values
d2 = d2.values
d = np.c_[d1,d2]
x=np.random.randn(len(c1),1)
e1 = x - d1 @ pinv(d1) @ x
e2 = x - d2 @ pinv(d2) @ x
e12 = e1 - d2 @ pinv(d2) @ e1
e21 = e2 - d1 @ pinv(d1) @ e2

d12 = d1 - d2 @ pinv(d2) @ d1
e22 = e2 - d12 @ pinv(d12) @ e2