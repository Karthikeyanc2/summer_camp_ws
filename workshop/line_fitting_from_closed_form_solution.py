import matplotlib
import numpy as np

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class LineFittingFromClosedFormSolution:
    def __init__(self):
        self.points = []
        self.current_line = None
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.ax.axis('off')
        self.ax.set_title('Line fitting linear')
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_aspect(1)
        self.fig.tight_layout()
        plt.show()

    def on_click(self, event):
        if event.xdata and event.ydata:
            self.points.append([event.xdata, event.ydata])
            self.ax.scatter(event.xdata, event.ydata, marker='x', color='green', linewidth=2)
            if len(self.points) > 1:
                m, c = self.get_line_parameters_m_and_c(self.points)
                x = np.array([-50, 50])
                y = m * x + c
                if self.current_line is not None:
                    self.current_line[0].remove()
                self.current_line = self.ax.plot(x, y, color='red', linewidth=2)
            self.fig.canvas.draw()

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


LineFittingFromClosedFormSolution()
