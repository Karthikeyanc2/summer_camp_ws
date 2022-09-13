import math

import matplotlib
import numpy as np
from numpy.random import default_rng

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

rng = default_rng()


class LineFittingRANSAC:
    def __init__(self, n=2, k=1000, t=1, d=2):
        # n=10, k=100, t=0.05, d=10
        self.n = n  # `n`: Minimum number of data points to estimate parameters
        self.k = k  # `k`: Maximum iterations allowed
        self.t = t  # `t`: Threshold value to determine if points are fit well
        self.d = d  # `d`: Number of close data points required to assert model fits well

        self.points = []
        self.current_line = None
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.ax.axis('off')
        self.ax.set_title('RANSAC line fitting')
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_aspect(1)
        self.fig.tight_layout()
        plt.show()

    def on_click(self, event):
        if event.xdata and event.ydata:
            self.points.append([event.xdata, event.ydata])
            colors = [True for _ in range(len(self.points))]
            if len(self.points) > self.d:
                (m, c), colors = self.fit_ransac(np.asarray(self.points))
                if not(m == 0 and c == 0):
                    x = np.array([-50, 50])
                    y = m * x + c
                    if self.current_line is not None:
                        self.current_line[0].remove()
                    self.current_line = self.ax.plot(x, y, color='blue', linewidth=2)
                else:
                    colors = [True for _ in range(len(self.points))]
            self.ax.scatter([x for x, y in self.points], [y for x, y in self.points], marker='x', color=["green" if c else "red" for c in colors], linewidth=2)
            self.fig.canvas.draw()

    def fit_ransac(self, points):
        """
        :param points: Nx2
        :return:
        """
        best_error = np.inf
        best_mc = 0, 0

        for _ in range(self.k):
            ids = rng.permutation(len(points))
            mc = self.get_line_parameters_m_and_c(points[ids[:self.n]])
            distances = self.get_distance_to_points_from_a_line(points[ids[self.n:]], mc)
            inlier_points = points[ids[self.n:]][distances < self.t]

            if len(inlier_points) > self.d:
                all_inlier_points = np.vstack([points[ids[:self.n]], inlier_points])
                mc = self.get_line_parameters_m_and_c(all_inlier_points)
                error = self.get_distance_to_points_from_a_line(all_inlier_points, mc).mean()
                if error < best_error:
                    best_error = error
                    best_mc = mc

        distances = self.get_distance_to_points_from_a_line(points, best_mc)
        return best_mc, distances < self.t

    @staticmethod
    def get_distance_to_points_from_a_line(points, mc):
        """
        :param points: list of points [[x1, y1], ... [xm, ym]]
        :param mc: m, c
        :return:
        """
        m, c = mc
        distances = []
        for x, y in points:
            distances.append(abs(m * x - y + c) / math.sqrt(m * m + 1))

        return np.asarray(distances)

    @staticmethod
    def get_line_parameters_m_and_c(points):
        """
        :param points: list of points [[x1, y1], ... [xm, ym]]
        :return: m, c
        """
        X_matrix = []
        Y_matrix = []
        for point in points:
            X_matrix.append([point[0], 1])
            Y_matrix.append([point[1]])

        X_matrix = np.asarray(X_matrix)  # m x 2
        Y_matrix = np.asarray(Y_matrix)  # m x 1
        theta = np.linalg.inv(X_matrix.T.dot(X_matrix)).dot(X_matrix.T.dot(Y_matrix))

        return theta.T[0]


LineFittingRANSAC()
