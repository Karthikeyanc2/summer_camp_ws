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
            self.points.append([round(event.xdata, 2), round(event.ydata, 2)])
            self.ax.scatter(event.xdata, event.ydata, marker='x', color='green', linewidth=2)
            if len(self.points) > 1:
                m, c = self.get_line_parameters_w_and_b(self.points)
                x = np.array([-50, 50])
                y = m * x + c
                if self.current_line is not None:
                    self.current_line[0].remove()
                self.current_line = self.ax.plot(x, y, color='red', linewidth=2)
            self.fig.canvas.draw()

    @staticmethod
    def get_line_parameters_w_and_b(points):
        """
        @param points: list of points [[x1, y1], ... [xm, ym]]
        @return: w, b
        """
        """
        How to construct a numpy matrix?
        np_matrix = np.asarray([[1, 2], [2, 3], [6, 7]])
        
        How to take a transpose of a numpy matrix?
        np.matrix_transpose = np_matrix.T
        
        How to take inverse of a numpy matrix?
        np_matrix_inverse = np.linalg.inv(np_matrix)
        
        How to take dot product of two numpy matrix?
        output = matrix_a.dot(matrix_b)
        """
        print("--------------------------------------------------")
        print("points :", points)
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
