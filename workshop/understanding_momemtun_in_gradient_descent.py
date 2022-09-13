import time

import matplotlib
import numpy as np
from matplotlib.widgets import Button, TextBox

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt, gridspec


class GradientDescent:
    def __init__(self):
        self.prev_points_plot = None
        self.prev_loss_plot = None
        self.starting_point_scatter = None
        self.current_line_plot = None
        self.all_previous_arrows = []
        self.starting_m = 0
        self.previous_delta_m = 0
        self.lr = 0

        self.fig = plt.figure(figsize=(7, 7.2))
        self.grid_spec = gridspec.GridSpec(43, 40, figure=self.fig)
        self.ax = self.fig.add_subplot(self.grid_spec[:40, :])

        ax_refresh_button = self.fig.add_subplot(self.grid_spec[41:, :10])
        self.refresh_button = Button(ax_refresh_button, "Refresh")

        ax_lr_textbox = self.fig.add_subplot(self.grid_spec[41:, 20:29])
        self.lr_textbox = TextBox(ax_lr_textbox, 'Learning Rate: ', hovercolor='0.975', initial='0.001')

        ax_submit_button = self.fig.add_subplot(self.grid_spec[41:, 30:])
        self.submit_button = Button(ax_submit_button, 'Submit')

        self.refresh_button.on_clicked(self.refresh_clicked)
        self.submit_button.on_clicked(self.submit_clicked)
        self.refresh_clicked(None)

        self.ax.axis('off')
        self.ax.set_title('Understanding momentum in gradient descent (shown: error plot)')

        self.fig.tight_layout()
        plt.show()

    def refresh_clicked(self, _):
        [arrow.remove() for arrow in self.all_previous_arrows]
        self.all_previous_arrows = []

        ms = np.linspace(-2.25, 1.75, 50)
        losses = self.get_loss(ms)

        if self.prev_loss_plot is not None:
            [plot.remove() for plot in self.prev_loss_plot]
        self.prev_loss_plot = self.ax.plot(ms, losses, color='orange')
        self.ax.set_xlim([min(ms) - 0.1, max(ms) + 0.1])
        self.ax.set_ylim([min(losses) - 1, max(losses) + 1])
        self.starting_m = ms[np.argmax(losses)]
        if self.starting_point_scatter is not None:
            self.starting_point_scatter.remove()
        self.starting_point_scatter = self.ax.scatter(self.starting_m, np.max(losses), marker='x', color='green')
        self.fig.canvas.draw()

    @staticmethod
    def get_loss(ms):
        return 0.22 + 0.15 * ms - 0.41 * ms ** 2 + 0.21 * ms ** 3 + 0.2718 * ms ** 4

    @staticmethod
    def get_gradient(ms):
        return 0.15 - 0.82 * ms + 0.63 * ms ** 2 + 1.0872 * ms ** 3

    def submit_clicked(self, _):
        [arrow.remove() for arrow in self.all_previous_arrows]
        self.all_previous_arrows = []
        self.lr = float(self.lr_textbox.text)
        new_m = self.starting_m
        loss = self.get_loss(new_m)
        prev_point = (new_m, loss)
        for _ in range(50):
            # new_m = self.get_next_m(new_m, self.lr)
            new_m = self.get_next_m(new_m, self.lr)
            loss = self.get_loss(new_m)
            self.starting_point_scatter.remove()
            self.starting_point_scatter = self.ax.scatter(new_m, loss, marker='x', color='green')
            self.all_previous_arrows.append(self.ax.arrow(
                prev_point[0], prev_point[1], new_m - prev_point[0], loss - prev_point[1], head_width=0.05,
                length_includes_head=True))
            prev_point = (new_m, loss)
            self.fig.canvas.draw()
            plt.pause(0.02)
        self.starting_m = new_m

    def get_next_m(self, old_m, lr):
        """
        :param old_m: float : current m
        :param lr: float: learning rate
        :return: float: new m
        """
        dl_by_dm = self.get_gradient(old_m)
        new_m = old_m - lr * dl_by_dm
        return new_m

    def get_next_m_with_momentum(self, old_m, lr, momentum=0.9):
        """
        :param old_m: float : current m
        :param lr: float: learning rate
        :param momentum: float: momentum
        :return: float: new m
        """
        dl_by_dm = self.get_gradient(old_m)
        current_delta_m = momentum * self.previous_delta_m - lr * dl_by_dm
        new_m = old_m + current_delta_m
        self.previous_delta_m = current_delta_m
        return new_m


GradientDescent()
