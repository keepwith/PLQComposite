import numpy as np
import matplotlib.pyplot as plt
from plqcom.base import rehu
from plqcom.base import relu

# right
x = np.linspace(0, 6, 1000)
y = np.array(relu(2 * x) + relu(6 * x - 6) + rehu(2 * (x - 1), 2) + relu(12 * x - 24))
plt.figure()
plt.plot(x, y)
x = np.linspace(0, 1, 1000)
y = np.array(2 * x)
plt.plot(x, y)
x = np.linspace(1, 2, 1000)
y = np.array(2 * x * x + 4 * x - 4)
plt.plot(x, y)
x = np.linspace(2, 4, 1000)
y = np.array(24 * x - 36)
plt.plot(x, y)

plt.xlim(-4, 6)
plt.show()

# left
x = np.linspace(-6, 0, 1000)
y = np.array(relu(-2 * x) + relu(-4 * x - 16) + rehu(-np.sqrt(2) * (x + 4), np.inf))
plt.figure()
plt.plot(x, y)
x = np.linspace(-4, 0, 1000)
y = np.array(-2 * x)
plt.plot(x, y)
x = np.linspace(-6, -4, 1000)
y = np.array(x * x + 2 * x)
plt.plot(x, y)


plt.xlim(-6, 6)
plt.show()
