import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[2],[4],[6],[8],[10]])
y = np.array([0,0,1,1,1])

model = LogisticRegression().fit(x,y)
print("Prediction:", model.predict([[5]]))
