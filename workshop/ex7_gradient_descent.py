import time

import matplotlib
import numpy as np
from matplotlib.widgets import Button, TextBox

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt, gridspec


class GradientDescent:
    def __init__(self):
        self.points = []
        self.prev_points_plot = None
        self.prev_loss_plot = None
        self.starting_point_scatter = None
        self.current_line_plot = None
        self.all_previous_arrows = []
        self.starting_m = 0
        self.lr = 0

        self.fig = plt.figure(figsize=(14, 7))
        self.grid_spec = gridspec.GridSpec(43, 80, figure=self.fig)
        self.ax_left = self.fig.add_subplot(self.grid_spec[:40, :40])
        self.ax_right = self.fig.add_subplot(self.grid_spec[:40, 40:])  # , projection='3d')

        ax_refresh_button = self.fig.add_subplot(self.grid_spec[41:, :20])
        self.refresh_button = Button(ax_refresh_button, "Refresh")

        ax_lr_textbox = self.fig.add_subplot(self.grid_spec[41:, 30:-22])
        self.lr_textbox = TextBox(ax_lr_textbox, 'Learning Rate: ', hovercolor='0.975', initial='0.001')

        ax_submit_button = self.fig.add_subplot(self.grid_spec[41:, -20:])
        self.submit_button = Button(ax_submit_button, 'Submit')

        self.refresh_button.on_clicked(self.refresh_clicked)
        self.submit_button.on_clicked(self.submit_clicked)
        self.refresh_clicked(None)

        self.ax_left.axis('off')
        self.ax_left.set_title('Line fitting')
        self.ax_left.set_xlim([-1, 1])
        self.ax_left.set_ylim([-1, 1])

        self.ax_right.axis('off')
        self.ax_right.set_title('Error plot')

        self.fig.tight_layout()
        plt.show()

    def refresh_clicked(self, _):
        [arrow.remove() for arrow in self.all_previous_arrows]
        self.all_previous_arrows = []
        x = np.random.random(30) * 2 - 1
        m = np.random.random(1)*4 - 2
        y = m * x + np.random.normal(0, 0.1, len(x))
        if self.prev_points_plot is not None:
            self.prev_points_plot.remove()
        self.prev_points_plot = self.ax_left.scatter(x, y, marker='x', color='blue')
        self.points = np.asarray([x, y])
        ms = np.tan(np.linspace(-np.pi/2 + 0.2, np.pi/2 - 0.2, 50))
        losses = np.power(self.points[1][None, :] - ((ms[:, None] * self.points[0][None, :])), 2).sum(axis=1)
        if self.prev_loss_plot is not None:
            [plot.remove() for plot in self.prev_loss_plot]
        self.prev_loss_plot = self.ax_right.plot(ms, losses, color='orange')
        self.ax_right.set_xlim([min(ms)-0.1, max(ms)+0.1])
        self.ax_right.set_ylim([min(losses)-1, max(losses)+1])
        self.starting_m = ms[np.argmax(losses)]
        if self.starting_point_scatter is not None:
            self.starting_point_scatter.remove()
        self.starting_point_scatter = self.ax_right.scatter(self.starting_m, np.max(losses), marker='x', color='green')
        self.draw_regression_line(self.starting_m)
        self.fig.canvas.draw()

    def draw_regression_line(self, m):
        x = np.array([-50, 50])
        y = m * x
        if self.current_line_plot is not None:
            self.current_line_plot[0].remove()
        self.current_line_plot = self.ax_left.plot(x, y, color='red', linewidth=2)

    def submit_clicked(self, _):
        [arrow.remove() for arrow in self.all_previous_arrows]
        self.all_previous_arrows = []
        self.lr = float(self.lr_textbox.text)
        new_m = self.starting_m
        loss = np.power(self.points[1] - new_m * self.points[0], 2).sum()
        prev_point = (new_m, loss)
        for _ in range(50):
            # grad = sum(2 * self.points[0] * (-self.points[1] + new_m * self.points[0]))
            # new_m = new_m - self.lr * grad
            new_m = self.get_next_w(np.asarray(self.points).T, new_m, self.lr)
            loss = np.power(self.points[1] - new_m * self.points[0], 2).sum()
            self.starting_point_scatter.remove()
            self.starting_point_scatter = self.ax_right.scatter(new_m, loss, marker='x', color='green')
            self.all_previous_arrows.append(self.ax_right.arrow(
                prev_point[0], prev_point[1], new_m - prev_point[0], loss - prev_point[1], head_width=0.05, length_includes_head=True))
            prev_point = (new_m, loss)
            self.draw_regression_line(new_m)
            self.fig.canvas.draw()
            plt.pause(0.02)
        self.starting_m = new_m

    @staticmethod
    def get_next_w(points, old_w, lr):
        """
        @param points: N x 2 [(x1, y1), ... (xn, yn)]
        @param old_w: float : current w
        @param lr: float: learning rate
        @return: float: new w
        """
        dl_by_dw = sum(2 * points[:, 0] * (old_w * points[:, 0] - points[:, 1]))
        new_w = old_w - lr * dl_by_dw
        return new_w


GradientDescent()
