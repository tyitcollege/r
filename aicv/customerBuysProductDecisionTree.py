import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier

x = np.array([[25,0],[30,1],[45,2],[35,1],[50,0]])
y = np.array([0,1,1,1,0])

model = DecisionTreeClassifier().fit(x,y)
print("Prediction:", model.predict([[40,2]]))

plt.figure(figsize=(10,6))
tree.plot_tree(model,feature_names=["Age","Income"],class_names=["No","Yes"],
filled=True)
plt.show()
