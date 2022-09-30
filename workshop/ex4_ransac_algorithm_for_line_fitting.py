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
        self.k = k  # `k`: Maximum iterations allowed
        self.n = n  # `n`: Number of points to sample randomly to determine the line-fit
        self.t = t  # `t`: Threshold to determine inlier points after a line-fit
        self.d = d  # `d`: Minimum number of inlier points required to say that the estimated line is probably a correct line

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
            self.points.append([round(event.xdata, 2), round(event.ydata, 2)])
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
        @param points: Nx2
        @return:
        """
        best_error = np.inf
        best_mc = 0, 0

        for _ in range(self.k):
            ids = rng.permutation(len(points))
            mc = self.get_line_parameters_w_and_b(points[ids[:self.n]])
            distances = self.get_distance_to_points_from_a_line(points[ids[self.n:]], mc)
            distances = np.asarray(distances)
            inlier_points = points[ids[self.n:]][distances < self.t]

            if len(inlier_points) > self.d:
                all_inlier_points = np.vstack([points[ids[:self.n]], inlier_points])
                mc = self.get_line_parameters_w_and_b(all_inlier_points)
                distances = self.get_distance_to_points_from_a_line(all_inlier_points, mc)
                error = np.asarray(distances).mean()
                if error < best_error:
                    best_error = error
                    best_mc = mc

        distances = self.get_distance_to_points_from_a_line(points, best_mc)
        distances = np.asarray(distances)
        return best_mc, distances < self.t

    @staticmethod
    def get_distance_to_points_from_a_line(points, wb):
        """
        @param points: list of points [[x1, y1], ... [xm, ym]]
        @param wb: w, b
        @return: distances [d1, d2, .... dm]
        """
        """
        How to take square root of a number?
        num = 5
        sqrt_of_5 = math.sqrt(5)
        """
        print("-------------------------------------------------")
        print("points", points)
        print("line parameters (w, b): ", wb)
        w, b = wb
        distances = []
        for x, y in points:
            distances.append(abs(w * x - y + b) / math.sqrt(w * w + 1))

        return distances

    @staticmethod
    def get_line_parameters_w_and_b(points):
        """
        @param points: list of points [[x1, y1], ... [xm, ym]]
        @return: w, b
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
