import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[1000],[1500],[2000],[2500],[3000]])
y = np.array([150000,200000,250000,300000,350000])

model = LinearRegression().fit(x,y)
predicted_price = model.predict([[2200]])
print("Predicted Price:", model.predict([[2200]]))

plt.scatter(x,y)
plt.plot(x, model.predict(x))
plt.scatter(2200, predicted_price)
plt.xlabel("Square footage")
plt.ylabel("Price")
plt.show()
