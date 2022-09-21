#!/usr/bin/env python3
import math

import rospy
import ros_numpy
from geometry_msgs.msg import Point, Quaternion, Vector3
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from sklearn.neighbors import LocalOutlierFactor
from vehicle_msgs.msg import AdmaData
from sensor_msgs.msg import PointCloud2
import pandas as pd
from matplotlib. path import Path as MatplotlibPath
import numpy as np
import pyclipper
import rospkg
from pathlib import Path
import tf
from dbscan import DBSCAN
from sklearn import linear_model
from trimesh.bounds import oriented_bounds_2D
from scipy.spatial.transform import Rotation


class Object:
    def __init__(self):
        self.locations = np.asarray([])  # (x, y)
        self.max_history_to_store = 5
        self.order = 3
        self.theta_x = 0
        self.theta_y = 0
        self.next_x = self.max_history_to_store

    def filter(self, location):
        self.locations = np.append(self.locations[(1 if len(self.locations) >= self.max_history_to_store else 0):], np.asarray([location]))
        self.next_x = self.max_history_to_store
        if len(self.locations) >= self.max_history_to_store:
            X_matrix = np.power(np.arange(self.max_history_to_store)[:, None], range(self.order + 1))
            # np.asarray([np.power(range(self.max_history_to_store), i) for i in range(self.order + 1)]).T
            self.theta_x = np.linalg.inv(X_matrix.T.dot(X_matrix)).dot(X_matrix.T.dot(self.locations[:, :0])).T[0]
            self.theta_y = np.linalg.inv(X_matrix.T.dot(X_matrix)).dot(X_matrix.T.dot(self.locations[:, :1])).T[0]
            return sum(X_matrix[-1] * self.theta_x), sum(X_matrix[-1] * self.theta_y)
        else:
            return location

    def predict(self):
        x_vector = np.power(self.next_x, range(self.order + 1))
        # np.asarray([np.power(self.last_x_used, i) for i in range(self.order + 1)]).T
        self.locations = np.append(self.locations[1:], np.asarray([sum(x_vector * self.theta_x), sum(x_vector * self.theta_y)]))
        self.next_x += 1
        return self.locations[-1]

    def get_angle(self, angle):
        x1, y1 = self.locations[-1]
        x2, y2 = self.locations[-2]
        vel = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 0.1
        if vel > 2:
            return math.atan2(y2 - y1, x2 - x1)
        else:
            return angle


class TTCEstimator:
    def __init__(self):
        self.adma_msg = None

        self.package_path = "/home/schenker2/karthik/summer_camp_ws/src/ttc_based_critical_situation_identifier"
        # rospkg.RosPack().get_path("ttc_based_critical_situation_identifier")
        track_data = np.array(pd.read_csv(Path(self.package_path) / "config/outdoor_contour_track_data_sp80.csv")[["x_relative", "y_relative"]])
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(track_data, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(-2)
        self.track_path = MatplotlibPath(solution[0])

        self.transformer = tf.TransformListener(True, rospy.Duration(20))
        self.transformer.setUsingDedicatedThread(True)

        self.ransac_regressor = linear_model.RANSACRegressor(residual_threshold=0.3, max_trials=100)
        self.lof_filter = LocalOutlierFactor(n_neighbors=2, p=1)

        self.points_publisher = rospy.Publisher('processed_points', PointCloud2, queue_size=10)
        self.bbox_publisher = rospy.Publisher('objects', BoundingBoxArray, queue_size=10)
        rospy.Subscriber("velodyne/points", PointCloud2, self.pcd_callback)
        # rospy.Subscriber("/vehicle_marie/adma_data", ADMAData, self.adma_callback)
        rospy.Subscriber("/vehicle_isaak/adma_data", AdmaData, self.adma_callback_isaak)
        self.latest_isaak_position = None
        self.isaak_detected_object = Object()
        self.previous_objects = []

    def adma_callback_isaak(self, msg):
        self.latest_isaak_position = msg

    @staticmethod
    def get_points_from_point_cloud(msg):
        array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        points = np.zeros((len(array), 4), dtype=np.float64)
        points[..., 0] = array['x']
        points[..., 1] = array['y']
        points[..., 2] = array['z']
        points[..., 3] = array['intensity']
        return points

    @staticmethod
    def get_point_cloud_from_points(points, original_msg):
        pts = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32)
        ])
        pts['x'], pts['y'], pts['z'], pts['intensity'] = points.transpose()
        pcl2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(pts, stamp=original_msg.header.stamp, frame_id=original_msg.header.frame_id)
        return pcl2_msg

    def pcd_callback(self, msg):
        basestation_T_velodyne = self.transformer.asMatrix("/basestation", msg.header)
        points = self.get_points_from_point_cloud(msg)
        points = self.roi_removal(points, basestation_T_velodyne)
        # points = self.z_based_ground_removal(points)
        points = self.ransac_based_ground_removal(points)
        points = self.outlier_filter(points)
        clusters = self.clustering(points)
        bboxes = self.object_detection_naive(clusters, msg.header)
        self.bbox_publisher.publish(bboxes)
        foreground_points_as_message = self.get_point_cloud_from_points(points, msg)
        self.points_publisher.publish(foreground_points_as_message)

    def roi_removal(self, points, transformation_matrix):
        all_points_in_vehicle_coordinates = np.ones_like(points)
        all_points_in_vehicle_coordinates[:, :3] = points[:, :3]
        all_points_in_global_coordinates = all_points_in_vehicle_coordinates.dot(transformation_matrix.T)
        points_inside_test_track_in_vehicle_coordinates = points[self.track_path.contains_points(all_points_in_global_coordinates[:, :2])]
        return points_inside_test_track_in_vehicle_coordinates

    @staticmethod
    def z_based_ground_removal(points):
        threshold = -1.5
        return points[points[:, 2] > threshold]

    def ransac_based_ground_removal(self, points):
        self.ransac_regressor.fit(points[:, :2], points[:, 2])
        outlier_mask = np.logical_not(self.ransac_regressor.inlier_mask_)
        return points[outlier_mask]

    def outlier_filter(self, points):
        mask = self.lof_filter.fit_predict(points)
        return points[mask == 1]

    @staticmethod
    def clustering(points):
        labels, core_samples_mask = DBSCAN(points[:, :2].astype(np.float64), eps=2, min_samples=2)
        unique_labels = set(labels)

        clusters = []
        for k in unique_labels:
            if k != -1:
                class_members_mask = labels == k
                this_cluster = points[class_members_mask & core_samples_mask]
                if len(this_cluster) > 5:
                    clusters.append(this_cluster)

        return clusters

    @staticmethod
    def object_detection_naive(clusters, header):
        boxes = []
        for cluster in clusters:
            box = BoundingBox()
            box.header = header

            z_min, z_max = cluster[:, 2].min(), cluster[:, 2].max()
            transformation_matrix_2d, (length, width) = oriented_bounds_2D(cluster[:, :2])
            # this transformation is local_T_global, but we need --> global_T_local
            print(transformation_matrix_2d[:-1, -1])
            transformation_matrix_2d = np.linalg.inv(transformation_matrix_2d)
            box.pose.position = Point(transformation_matrix_2d[0, -1], transformation_matrix_2d[1, -1], (z_max + z_min) / 2)
            print(box.pose.position)
            rotation_matrix_3d = transformation_matrix_2d
            rotation_matrix_3d[:2, -1] = 0
            box_orientation = Rotation.from_matrix(rotation_matrix_3d).as_quat()
            box.pose.orientation = Quaternion(*box_orientation)
            box.dimensions = Vector3(length, width, z_max - z_min)
            boxes.append(box)
        print(len(clusters))

        bboxes = BoundingBoxArray()
        bboxes.header = header
        bboxes.boxes = boxes
        return bboxes

    def object_detection(self, clusters, header):
        isaak_box = None

        for cluster in clusters:
            z_min, z_max = cluster[:, 2].min(), cluster[:, 2].max()
            transformation_matrix_2d, (_, _) = oriented_bounds_2D(cluster[:, :2])
            transformation_matrix_2d = np.linalg.inv(transformation_matrix_2d)
            position = transformation_matrix_2d[0, -1], transformation_matrix_2d[1, -1], (z_max + z_min) / 2

            if distance(position, self.latest_isaak_position) < 3:
                box = BoundingBox()
                box.header = header
                orientation = math.atan2(transformation_matrix_2d[1, 0], transformation_matrix_2d[0, 0])

                filtered_position = self.isaak_detected_object.filter([position[0], position[1]])
                filtered_angle = self.isaak_detected_object.get_angle(orientation)

                rotation_matrix = np.array([[math.cos(-filtered_angle), math.sin(-filtered_angle)],
                                            [-math.sin(-filtered_angle), math.cos(-filtered_angle)]])
                rotated_cluster_abs = np.abs((cluster[:, :2] - np.asarray(filtered_position).reshape(1, 2)).dot(rotation_matrix))
                length, width = np.abs(rotated_cluster_abs[:, 0]).max() * 2, np.abs(rotated_cluster_abs[:, 0]).min() * 2
                length, width = max(length, width), min(length, width)
                dimensions = length, width, z_max - z_min
                isaak_box = box




    def ttc_estimator(self, cluster, adma_information):
        pass
        # ml inference
        # return ttc


rospy.init_node('blas')
TTCEstimator()
rospy.spin()
