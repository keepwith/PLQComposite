import numpy as np
import matplotlib.pyplot as plt
from rehline import relu, rehu


def plot_origin():
    plt.figure()
    x = np.linspace(0, 1, 1000)
    y = np.array(2 * x)
    plt.plot(x, y)
    x = np.linspace(1, 2, 1000)
    y = np.array(2 * x * x + 4 * x - 4)
    plt.plot(x, y)
    x = np.linspace(2, 6, 1000)
    y = np.array(24 * x - 36)
    plt.plot(x, y)

    x = np.linspace(-4, 0, 1000)
    y = np.array(-2 * x)
    plt.plot(x, y)
    x = np.linspace(-6, -4, 1000)
    y = np.array(x * x + 2 * x)
    plt.plot(x, y)
    plt.xlabel('z')
    plt.ylabel('L(z)')
    plt.legend(['PLQ(z)'])
    plt.title('PLQ Loss ')
    plt.xlim(-6, 6)
    plt.savefig('../figs/PLQLoss.png')


def plot_left_right():
    plt.figure()
    x_0 = np.linspace(-6, 0, 1000)
    y_0 = np.array(0 * x_0)
    x_1 = np.linspace(0, 1, 1000)
    y_1 = np.array(2 * x_1)
    x_2 = np.linspace(1, 2, 1000)
    y_2 = np.array(2 * x_2 * x_2 + 4 * x_2 - 4)
    x_3 = np.linspace(2, 6, 1000)
    y_3 = np.array(24 * x_3 - 36)
    x = np.concatenate((x_0, x_1, x_2, x_3))
    y = np.concatenate((y_0, y_1, y_2, y_3))
    plt.plot(x, y)

    x_0 = np.linspace(-6, -4, 1000)
    y_0 = np.array(x_0 * x_0 + 2 * x_0)
    x_1 = np.linspace(-4, 0, 1000)
    y_1 = np.array(-2 * x_1)
    x_2 = np.linspace(0, 6, 1000)
    y_2 = np.array(0 * x_3)
    x = np.concatenate((x_0, x_1, x_2))
    y = np.concatenate((y_0, y_1, y_2))
    plt.plot(x, y)

    plt.xlabel('z')
    plt.ylabel('L(z)')
    plt.legend(['Right', 'Left'])
    plt.title('Separate PLQ Loss to left and right')
    plt.savefig('../figs/Left_Right.png')


def plot_relu_rehu():
    # ReLU-ReHU Loss Decomposition Result
    plt.subplot(1, 2, 1)
    # right
    x_1 = np.linspace(0, 6, 1000)
    y_1 = np.array(relu(2 * x_1) + relu(6 * x_1 - 6) + rehu(2 * (x_1 - 1), 2) + relu(12 * x_1 - 24))
    plt.figure()

    # left
    x_2 = np.linspace(-6, 0, 1000)
    y_2 = np.array(relu(-2 * x_2) + relu(-4 * x_2 - 16) + rehu(-np.sqrt(2) * (x_2 + 4), np.inf))
    plt.figure()
    x = np.append(x_2, x_1)
    y = np.append(y_2, y_1)
    plt.plot(x, y)
    plt.xlim(-6, 6)
    plt.xlabel('z')
    plt.ylabel('L(z)')
    plt.legend(['ReLU-ReHU(z)'])
    plt.title('ReLU-ReHU Loss Decomposition Result')
    plt.savefig('../figs/ReLU_ReHU.png')


if __name__ == '__main__':
    plot_relu_rehu()
