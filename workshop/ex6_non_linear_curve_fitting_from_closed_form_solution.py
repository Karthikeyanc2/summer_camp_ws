import math
import numpy as np

import matplotlib
from matplotlib.widgets import Slider

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import gridspec

plt.rcParams['text.usetex'] = True


class NonLinearCurveFittingFromClosedFormSolution:
    def __init__(self):
        self.points = []
        self.current_line = None

        self.fig = plt.figure(constrained_layout=True)
        self.grid_spec = gridspec.GridSpec(43, 40, figure=self.fig)
        self.ax = self.fig.add_subplot(self.grid_spec[:40, :])
        slider_axis = self.fig.add_subplot(self.grid_spec[41:43, :])
        self.slider = Slider(slider_axis, "Polynomial order: ", 1, 10, valstep=1)
        self.slider.on_changed(self.on_slider_changed)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.ax.axis('off')
        self.ax.set_title('Non linear curve fitting')
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_aspect(1)
        self.fig.tight_layout()
        plt.show()

    def on_slider_changed(self, _):
        if len(self.points) > 1:
            order = int(self.slider.val)
            theta = np.asarray(self.get_curve_parameters(self.points, order)).reshape(-1)
            x = np.linspace(-10, 10, 100)
            y = np.asarray([sum(np.asarray([np.power(px, i) for i in range(order + 1)]) * theta) for px in x])
            if self.current_line is not None:
                self.current_line[0].remove()
            self.current_line = self.ax.plot(x, y, color='red', linewidth=2)
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes is self.ax:
            current_point = [round(event.xdata, 2), round(event.ydata, 2)]
            self.points.append(current_point)
            self.ax.scatter(event.xdata, event.ydata, marker='x', color='green', linewidth=2)
            self.on_slider_changed(None)

    @staticmethod
    def get_curve_parameters(points, order):
        """
        @param points: list of points [[x1, y1], ... [xm, ym]]
        @param order: a number (int)
        @return: theta [theta1, theta2, ...] (length should be equal to polynomial-order + 1)
        Example: for order 2 --> y = [theta1, theta2, theta3] . [1, x, xx]
        """
        print('-----------------------------------------------')
        print("points: ", points)
        print("polynomial order: ", order)
        X_matrix = []
        Y_matrix = []

        for point in points:
            this_point_polynomial = []
            for i in range(order + 1):
                this_point_polynomial.append(np.power(point[0], i))

            X_matrix.append(this_point_polynomial)
            Y_matrix.append([point[1]])

        X_matrix = np.asarray(X_matrix)  # m x order+1
        Y_matrix = np.asarray(Y_matrix)  # m x 1
        theta = np.linalg.inv(X_matrix.T.dot(X_matrix)).dot(X_matrix.T.dot(Y_matrix))

        return theta.T[0]


NonLinearCurveFittingFromClosedFormSolution()
