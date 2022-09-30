import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random


class PointInPolygon:
    def __init__(self, outdoor_polygon_flag=False):
        outdoor_polygon = pd.read_csv("../src/ttc_based_critical_situation_identifier/asset/outdoor_contour_track_data_sp80.csv")[["x_relative", "y_relative"]]

        if outdoor_polygon_flag:
            self.polygon = np.round(outdoor_polygon.to_numpy(), 2).tolist()[:-1]
        else:
            self.polygon = [[1, 2], [1.3, 10.6], [15, 10], [5, 6]]

        self.points = []
        self.fig, self.ax = plt.subplots()
        self.ax.plot([p[0] for p in self.polygon] + [self.polygon[0][0]],
                     [p[1] for p in self.polygon] + [self.polygon[0][1]], color="blue", linewidth=2)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        min_x = min([p[0] for p in self.polygon]) - 4
        min_y = min([p[1] for p in self.polygon]) - 4
        max_x = max([p[0] for p in self.polygon]) + 4
        max_y = max([p[1] for p in self.polygon]) + 4
        self.ax.plot([min_x, min_x, max_x, max_x, min_x], [min_y, max_y, max_y, min_y, min_y], color="black")
        self.ax.axis('off')
        self.ax.set_title('Point in polygon visualization')
        self.ax.set_aspect(1)
        self.fig.tight_layout()
        if outdoor_polygon_flag:
            self.load_one_frame_pcd()
        plt.show()

    def load_one_frame_pcd(self):
        points = np.round(np.loadtxt("../src/ttc_based_critical_situation_identifier/asset/one_complete_frame.csv"), 2).tolist()
        for point in points:
            color = "green" if self.point_in_polygon(self.polygon, point) else "red"
            self.ax.scatter(point[0], point[1], linewidths=2, color=color, marker='x')

    def on_click(self, event):
        point = [event.xdata, event.ydata]
        self.points.append(point)
        self.ax.scatter(point[0], point[1], linewidths=2, color="green" if self.point_in_polygon(self.polygon, point) else "red", marker='x')
        self.fig.canvas.draw()

    @staticmethod
    def point_in_polygon(polygon, point):
        """
        @param polygon: polygon defined as a list of points; [[x0, y0], [x1, y1], ...]
        @param point: point to check if it is inside polygon; [xp, yp]
        return: True if the point is in polygon else False
        """
        """
        How to loop through points?
        for i in range(len(points)):
            point = points[i]
        """
        print("------------------------------------------------------------------")
        print("polygon [[x0, y0], [x1, y1], ...] : ", polygon)
        print("point [xp, yp] :", point)

        count = 0

        xp = point[0]
        yp = point[1]

        for i in range(len(polygon)):
            x1 = polygon[i - 1][0]
            x2 = polygon[i][0]
            y1 = polygon[i - 1][1]
            y2 = polygon[i][1]

            # check if exactly one point is above and one point is below the line y = yp
            check_1 = (y2 > yp) != (y1 > yp)

            # check if the line made by polygon[i] and polygon[i-1] crosses the line y = yp between [xp, inf]
            check_2 = ((yp - y1) * (x2 - x1) / (y2 - y1) + x1) >= xp

            if check_1 and check_2:
                count += 1

        # If the number of crossings was odd, the point is in the polygon
        return count % 2 != 0


PointInPolygon()
PointInPolygon(outdoor_polygon_flag=True)
