import math
import numpy as np

import matplotlib
from matplotlib.widgets import Slider

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import gridspec

plt.rcParams['text.usetex'] = True


class CoordinateTransformation:
    def __init__(self):
        self.fig = plt.figure(constrained_layout=True)
        self.grid_spec = gridspec.GridSpec(46, 45, figure=self.fig)
        self.ax = self.fig.add_subplot(self.grid_spec[:40, :])
        slider_x_axis = self.fig.add_subplot(self.grid_spec[40:42, :])
        self.slider_x = Slider(slider_x_axis, r"$t_x$", -5, 5, 2)
        self.slider_x.on_changed(self.update_slider_values)

        slider_y_axis = self.fig.add_subplot(self.grid_spec[42:44, :])
        self.slider_y = Slider(slider_y_axis, r"$t_y$", -5, 5, 0)
        self.slider_y.on_changed(self.update_slider_values)

        slider_theta_axis = self.fig.add_subplot(self.grid_spec[44:, :])
        self.slider_theta = Slider(slider_theta_axis, r"$\theta$", -math.pi, math.pi, 0)
        self.slider_theta.on_changed(self.update_slider_values)

        self.arrow_head_width = 0.05
        self.ax.arrow(0, 0, 1, 0, color="black", head_width=self.arrow_head_width, length_includes_head=True)
        self.ax.text(1, 0, r"${}^Ex$")
        self.ax.arrow(0, 0, 0, 1, color="black", head_width=self.arrow_head_width, length_includes_head=True)
        self.ax.text(0, 1, r"${}^Ey$")
        self.ax.text(-0.1, -0.1, r"$\{E\}$")

        self.on_slider_change_stuff = []
        self.on_click_change_stuff = []
        self.last_clicked_point = [2.5, 0.5]

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.update_slider_values(None)

        self.ax.axis('off')
        self.ax.set_title('Coordinate transformation visualization')
        self.ax.set_aspect(1)
        self.fig.tight_layout()
        plt.show()

    def update_slider_values(self, _):
        for item in self.on_slider_change_stuff:
            item.remove()
        self.on_slider_change_stuff = []
        tx, ty, theta = self.slider_x.val, self.slider_y.val, self.slider_theta.val
        self.on_slider_change_stuff.append(self.ax.arrow(tx, ty, math.cos(theta), math.sin(theta), color="red", head_width=self.arrow_head_width, length_includes_head=True))
        self.on_slider_change_stuff.append(self.ax.text(tx + math.cos(theta), ty + math.sin(theta), r"${}^Vx$"))
        self.on_slider_change_stuff.append(self.ax.arrow(tx, ty, -math.sin(theta), math.cos(theta), color="red", head_width=self.arrow_head_width, length_includes_head=True))
        self.on_slider_change_stuff.append(self.ax.text(tx - math.sin(theta), ty + math.cos(theta), r"${}^Vy$"))
        self.on_slider_change_stuff.append(self.ax.text(tx-0.1, ty-0.1, r"$\{V\}$"))
        self.on_click(None)

    def on_click(self, event):
        for item in self.on_click_change_stuff:
            item.remove()
        self.on_click_change_stuff = []
        if event is not None:
            if event.inaxes is self.ax:
                self.last_clicked_point = [event.xdata, event.ydata]
        vx, vy, theta = self.slider_x.val, self.slider_y.val, self.slider_theta.val
        px, py = self.last_clicked_point  # in global coordinates
        self.on_click_change_stuff.append(self.ax.arrow(vx, vy, px - vx, py - vy, color="black", linestyle='-.', head_width=self.arrow_head_width, length_includes_head=True))
        self.on_click_change_stuff.append(self.ax.text(px+0.04, py+0.04, r"$\mathbf{\textit{p}}$"))
        self.on_click_change_stuff.append(self.ax.text((px + vx)/2 + 0.04, (py + vy)/2 - 0.04, r"${}^V\mathbf{\textit{p}}$"))
        point_in_vehicle_coordinates_x = math.cos(theta) * px + math.sin(theta) * py - math.cos(theta) * vx - math.sin(theta) * vy
        point_in_vehicle_coordinates_y = -math.sin(theta) * px + math.cos(theta) * py + math.sin(theta) * vx - math.cos(theta) * vy

        point_in_x_axis = math.cos(theta) * point_in_vehicle_coordinates_x + vx, math.sin(theta) * point_in_vehicle_coordinates_x + vy
        self.on_click_change_stuff.append(self.ax.arrow(px, py, point_in_x_axis[0] - px, point_in_x_axis[1] - py, color="black",linestyle=':', head_width=self.arrow_head_width, length_includes_head=True))
        # self.on_click_change_stuff.append(self.ax.text((vx + point_in_x_axis[0])/2, (vy + point_in_x_axis[1])/2, r"${}^V\mathbf{\textit{p}}_x$"))

        point_in_y_axis = - math.sin(theta) * point_in_vehicle_coordinates_y + vx, math.cos(theta) * point_in_vehicle_coordinates_y + vy
        self.on_click_change_stuff.append(self.ax.arrow(px, py, point_in_y_axis[0] - px, point_in_y_axis[1] - py, color="black", linestyle=':', head_width=self.arrow_head_width, length_includes_head=True))
        # self.on_click_change_stuff.append(self.ax.text((vx + point_in_y_axis[0]) / 2, (vy + point_in_y_axis[1]) / 2, r"${}^V\mathbf{\textit{p}}_y$"))

        point_in_global_coordinates = self.convert_point_from_vehicle_coordinates_to_global_coordinates([vx, vy], theta, [point_in_vehicle_coordinates_x, point_in_vehicle_coordinates_y])
        self.on_click_change_stuff.append(self.ax.arrow(0, 0, point_in_global_coordinates[0], point_in_global_coordinates[1], color="black", linestyle='-.', head_width=self.arrow_head_width, length_includes_head=True))
        correct = (round(px, 2) == round(point_in_global_coordinates[0], 2)) and (round(py, 2) == round(point_in_global_coordinates[1], 2))
        self.on_click_change_stuff.append(self.ax.scatter(px, py, color="green" if correct else "red", linewidth=2, marker='o', s=50))
        self.on_click_change_stuff.append(self.ax.scatter(point_in_global_coordinates[0], point_in_global_coordinates[1], color="brown", linewidth=2, marker='x'))
        self.on_click_change_stuff.append(self.ax.text(point_in_global_coordinates[0] / 2 + 0.04, point_in_global_coordinates[1] / 2 - 0.04, r"${}^E\mathbf{\textit{p}}$"))
        self.fig.canvas.draw()

    @staticmethod
    def convert_point_from_vehicle_coordinates_to_global_coordinates(
            vehicle_coordinates, vehicle_orientation, point_in_vehicle_coordinates):
        """
        :param vehicle_coordinates: vehicle's current coordinates [x, y]
        :param vehicle_orientation: vehicle's current orientation theta
        :param point_in_vehicle_coordinates: point measured in vehicle coordinates [x, y]
        :return: measured point in global coordinates [x, y]
        """
        # return [1, 1]
        tx, ty = vehicle_coordinates
        theta = vehicle_orientation
        x, y = point_in_vehicle_coordinates

        return math.cos(theta) * x - math.sin(theta) * y + tx, \
               math.sin(theta) * x + math.cos(theta) * y + ty


CoordinateTransformation()
