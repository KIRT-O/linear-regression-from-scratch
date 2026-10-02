import numpy as np
import matplotlib.pyplot as plt


X = np.array([50, 70, 90, 110, 130]) # the feature
y = np.array([60, 75, 95, 115, 135]) # the label

# setting initial weight and bias
w = 1 
b = 0

learning_rate = 0.0001

for i in range(10000):

    # setting the prediction 
    y_pred = w * X + b

    loss = y - y_pred
    mse = np.mean(loss ** 2)

    if i % 1000 == 0:
        print("iteration:", i)
        print("actual:", y)
        print("prediction: ", y_pred)
        print("loss: ", loss)
        print("mse: ", mse)

    # gradients
    n = len(X)

    dw = (2 / n) * np.sum(X * (y_pred - y))
    db = (2 / n) * np.sum(y_pred - y)

    # update
    w = w - learning_rate * dw
    b = b - learning_rate * db
    if i % 1000 == 0:
        print("new w:", w)
        print("new b:", b)
        print("-----------------------------------------")


plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, label="Prediction")

plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()

plt.show()