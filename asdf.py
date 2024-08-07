import numpy as np
import matplotlib.pyplot as plt


W = np.random.randn(2, 2)
b = np.random.randn(1, 2)


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

x_min, x_max = -3, 3
y_min, y_max = -3, 3
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = softmax(np.dot(grid_points, W) + b)


predictions = np.argmax(Z, axis=1)


plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, predictions.reshape(xx.shape), cmap=plt.cm.RdBu, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.colorbar()
plt.scatter(grid_points[:, 0], grid_points[:, 1], c=predictions, cmap=plt.cm.RdBu, marker='.', alpha=0.1)
plt.show()
