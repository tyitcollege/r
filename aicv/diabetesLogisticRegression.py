import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[120,45],[85,30],[150,50],[70,25],[95,35],[180,55]])
y = np.array([1,0,1,0,0,1])

model = LogisticRegression().fit(x,y)
print("Prediction:", model.predict([[200,40]]))
print("Probability:", model.predict_proba([[200,40]]))
