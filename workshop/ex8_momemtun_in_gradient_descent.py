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
        self.previous_delta_w = 0
        self.lr = 0
        self.momentum = 0

        self.fig = plt.figure(figsize=(7, 7.2))
        self.grid_spec = gridspec.GridSpec(43, 40, figure=self.fig)
        self.ax = self.fig.add_subplot(self.grid_spec[:40, :])

        ax_refresh_button = self.fig.add_subplot(self.grid_spec[41:, :5])
        self.refresh_button = Button(ax_refresh_button, "Refresh")

        ax_lr_textbox = self.fig.add_subplot(self.grid_spec[41:, 8:13])
        self.lr_textbox = TextBox(ax_lr_textbox, 'LR: ', hovercolor='0.975', initial='0.001')

        ax_momentum_textbox = self.fig.add_subplot(self.grid_spec[41:, 21:25])
        self.momentum_textbox = TextBox(ax_momentum_textbox, 'Momentum: ', hovercolor='0.975', initial='0.0')

        ax_submit_button = self.fig.add_subplot(self.grid_spec[41:, 26:])
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
        self.momentum = float(self.momentum_textbox.text)
        new_m = self.starting_m
        loss = self.get_loss(new_m)
        prev_point = (new_m, loss)
        for _ in range(50):
            # new_m = self.get_next_m(new_m, self.lr)
            new_m = self.get_next_w_with_momentum(new_m, self.lr, self.momentum)
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

    def get_next_w_with_momentum(self, old_w, lr, momentum):
        """
        @param old_w: float : current w
        @param lr: float: learning rate
        @param momentum: float: momentum
        @return: float: new w
        """
        """
        self.get_gradient(old_w) can be used to compute dl_by_dw
        """
        return old_w


GradientDescent()
