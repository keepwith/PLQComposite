import numpy as np
import matplotlib.pyplot as plt
from holoviews.plotting.bokeh.styles import font_size

from rehline._loss import _relu, _rehu


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
    y_1 = np.array(_relu(2 * x_1) + _relu(6 * x_1 - 6) + _rehu(2 * (x_1 - 1), 2) + _relu(12 * x_1 - 24))
    plt.figure()

    # left
    x_2 = np.linspace(-6, 0, 1000)
    y_2 = np.array(_relu(-2 * x_2) + _relu(-4 * x_2 - 16) + _rehu(-np.sqrt(2) * (x_2 + 4), np.inf))
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


def plot_hinge_square():
    z = np.linspace(-2, 2, 100)
    L1 = np.maximum(1 - z, 0)
    L2 = 0.5 * (1 - z) ** 2
    font1 = {'family': 'Arial',
             # 'weight': 'normal',
             'size': 20,
             }
    plt.plot(z, L1, marker='o', label='Hinge Loss', color='#FDEBAA')
    plt.plot(z, L2, marker='s', label='Square Loss', color='#DBE4FB')
    plt.plot(z, np.maximum(L1, L2), marker='^', label='PLQ Loss', color='#ABD1BC')
    plt.legend(prop=font1)
    plt.xlabel('z', fontdict=font1)
    plt.ylabel('L(z)', fontdict=font1)
    plt.title('Hinge Loss and Square Loss', fontdict=font1)
    plt.xticks(size=20, fontproperties='Arial')
    plt.yticks(size=20, fontproperties='Arial')

    return plt

def plot_fl_fr():
    z = np.linspace(-2, 2, 100)

    def f_l(x):
        return np.where(x <= -1, 0.5 * (x - 1) ** 2,
                        np.where(x <= 1, 1 - x,
                                 0))
    def f_r(x):
        return np.where(x >= 1, 0.5 * (x - 1) ** 2, 0)
    font1 = {'family': 'Arial',
             # 'weight': 'normal',
             'size': 20,
             }
    plt.plot(z, f_l(z), marker='s', label='f_l', color='#08306B')
    plt.plot(z, f_r(z), marker='o', label='f_r', color='#EDB120')
    plt.legend(prop=font1)
    plt.xlabel('z', fontdict=font1)
    plt.ylabel('L(z)', fontdict=font1)
    plt.xticks(size=20, fontproperties='Arial')
    plt.yticks(size=20, fontproperties='Arial')
    plt.title('Decompose to Left and Right', fontdict=font1)

    return plt

def plot_relu_rehu_l():
    z = np.linspace(-2, 2, 100)
    z1 = np.linspace(-2, 2, 80)
    z2 = np.linspace(-2, 2, 60)
    z3 = np.linspace(-2, 2, 40)
    def f_l(x):
        return np.where(x <= -1, 0.5 * (x - 1) ** 2,
                        np.where(x <= 1, 1 - x,
                                 0))

    relu_1 = _relu(1 - z1)
    relu_2 = _relu(-1 - z2)
    rehu_1 = _rehu(-1 - z3, np.inf)
    font1 = {'family': 'Arial',
             # 'weight': 'normal',
             'size': 20,
             }
    plt.plot(z, f_l(z), marker='s', label='f_l', color='#08306B')
    plt.plot(z1, relu_1, marker='^', label='ReLU 1', color='#2171B5')
    plt.plot(z2, relu_2, marker='v', label='ReLU 2', color='#6BAED6')
    plt.plot(z3, rehu_1, marker='o', label='ReHU 1', color='#C6DBEF')
    plt.legend(prop=font1)
    plt.xlabel('z', fontdict=font1)
    plt.ylabel('L(z)', fontdict=font1)
    plt.xticks(size=20, fontproperties='Arial')
    plt.yticks(size=20, fontproperties='Arial')
    plt.title('Decompose of the left', fontdict=font1)

    return plt

def plot_relu_rehu_r():
    z = np.linspace(-2, 2, 100)
    z1 = np.linspace(-2, 2, 50)
    def f_r(x):
        return np.where(x >= 1, 0.5 * (x - 1) ** 2, 0)

    rehu_2 = _rehu(-1 + z1, np.inf)
    font1 = {'family': 'Arial',
             # 'weight': 'normal',
             'size': 20,
             }
    plt.plot(z, f_r(z), marker='o', label='f_r', color='#EDB120')
    plt.plot(z1, rehu_2, marker='s', label='ReHU 2', color='#FED976')
    plt.legend(prop=font1)
    plt.xlabel('z', fontdict=font1)
    plt.ylabel('L(z)', fontdict=font1)
    plt.xticks(size=20, fontproperties='Arial')
    plt.yticks(size=20, fontproperties='Arial')
    plt.title('Decompose of the right', fontdict=font1)

    return plt

if __name__ == '__main__':
    plt.figure(figsize=(16, 12), dpi=1200)
    plt.subplot(2, 2, 1)
    plot_hinge_square()
    plt.subplot(2, 2, 2)
    plot_fl_fr()
    plt.subplot(2, 2, 3)
    plot_relu_rehu_l()
    plt.subplot(2, 2, 4)
    plot_relu_rehu_r()
    plt.savefig('../figs/PLQ.png', bbox_inches='tight', dpi=1200)
