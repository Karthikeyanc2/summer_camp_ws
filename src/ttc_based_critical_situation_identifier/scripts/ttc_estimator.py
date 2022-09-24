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
    def __init__(self, x, y, z, orientation):
        self.locations = np.asarray([[x, y, z]]).reshape(-1, 3)  # (x, y)
        self.filtered_locations = np.asarray([[x, y, z]]).reshape(-1, 3)  # (x, y)
        self.max_history_to_store = 15
        self.order = 2
        self.X_matrix = np.power(np.arange(self.max_history_to_store)[:, None], range(self.order + 1))
        self.theta_x = 0
        self.theta_y = 0
        self.theta_z = 0
        self.theta_orientation = 0
        self.theta_speed = 0
        self.next_x = self.max_history_to_store
        self.previous_orientation = orientation
        self.previous_speed = 0
        self.width, self.length, self.height = 0, 0, 0
        self.max_width, self.max_length, self.max_height = 1.75, 2.75, 1.5

    def filter_position(self, x, y, z):
        self.locations = np.r_[self.locations[(1 if len(self.locations) >= self.max_history_to_store else 0):], np.asarray([x, y, z]).reshape(1, 3)]
        self.next_x = self.max_history_to_store
        if len(self.locations) >= self.max_history_to_store:
            self.theta_x = np.linalg.inv(self.X_matrix.T.dot(self.X_matrix)).dot(self.X_matrix.T.dot(self.locations[:, 0].reshape(-1, 1))).T[0]
            self.theta_y = np.linalg.inv(self.X_matrix.T.dot(self.X_matrix)).dot(self.X_matrix.T.dot(self.locations[:, 1].reshape(-1, 1))).T[0]
            self.theta_z = np.linalg.inv(self.X_matrix.T.dot(self.X_matrix)).dot(self.X_matrix.T.dot(self.locations[:, 2].reshape(-1, 1))).T[0]
            filtered_location = sum(self.X_matrix[-1] * self.theta_x), sum(self.X_matrix[-1] * self.theta_y), sum(self.X_matrix[-1] * self.theta_z)
            self.filtered_locations = np.r_[self.filtered_locations, np.asarray(filtered_location).reshape(1, 3)]
            return filtered_location
        else:
            filtered_location = x, y, z
            self.filtered_locations = np.r_[self.filtered_locations, np.asarray(filtered_location).reshape(1, 3)]
            return filtered_location

    def predict_position(self):
        x_vector = np.power(self.next_x, range(self.order + 1))
        self.locations = np.append(self.locations[1:], np.asarray([sum(x_vector * self.theta_x), sum(x_vector * self.theta_y)]))
        self.next_x += 1
        return self.locations[-1]

    def filter_orientation_and_speed(self, angle):

        if len(self.filtered_locations) >= self.max_history_to_store:
            self.filtered_locations = self.filtered_locations[-self.max_history_to_store:]
            delta_x = np.gradient(self.filtered_locations[:, 0])
            delta_y = np.gradient(self.filtered_locations[:, 1])
            speeds = np.sqrt(delta_y**2 + delta_x**2) / 0.1
            self.theta_speed = np.linalg.inv(self.X_matrix.T.dot(self.X_matrix)).dot(self.X_matrix.T.dot(speeds.reshape(-1, 1))).T[0]
            calculated_speed = sum(self.X_matrix[-1]*self.theta_speed)

            orientations = np.arctan2(delta_y, delta_x)
            self.theta_orientation = np.linalg.inv(self.X_matrix.T.dot(self.X_matrix)).dot(self.X_matrix.T.dot(orientations.reshape(-1, 1))).T[0]

            if calculated_speed > 2:
                self.previous_speed = calculated_speed
                self.previous_orientation = sum(self.X_matrix[-1]*self.theta_orientation)

            return self.previous_orientation, calculated_speed
        else:
            return angle, 0

    def filter_dimensions(self, l, w, h):
        self.length = min(max(self.length, max(l, w)), self.max_length)
        self.width = min(max(self.width, min(l, w)), self.max_width)
        self.height = min(max(self.height, h), self.max_height)
        return self.length, self.width, self.height


class TTCEstimator:
    def __init__(self):
        self.adma_msg = None
        self.package_path = rospkg.RosPack().get_path("ttc_based_critical_situation_identifier")
        track_data = np.array(pd.read_csv(Path(self.package_path) / "config/outdoor_contour_track_data_sp80.csv")[["x_relative", "y_relative"]])
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(track_data, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(-2)
        self.track_path = MatplotlibPath(solution[0])

        self.transformer = tf.TransformListener(True, rospy.Duration(20))
        self.transformer.setUsingDedicatedThread(True)

        self.ransac_regressor = linear_model.RANSACRegressor(residual_threshold=0.35, max_trials=200)
        self.lof_filter = LocalOutlierFactor(n_neighbors=2, p=1)

        self.points_publisher = rospy.Publisher('processed_points', PointCloud2, queue_size=10)
        self.bbox_publisher = rospy.Publisher('objects', BoundingBoxArray, queue_size=10)
        rospy.Subscriber("velodyne/points", PointCloud2, self.pcd_callback)
        rospy.Subscriber("/vehicle_marie/adma_data", AdmaData, self.adma_callback_marie)
        isaak_adma_msg = rospy.wait_for_message("/vehicle_isaak/adma_data", AdmaData)
        self.latest_marie_position_and_velocity = [0, 0, 0], 0
        x, y, z = isaak_adma_msg.pose_cg.position.x, isaak_adma_msg.pose_cg.position.y, isaak_adma_msg.pose_cg.position.z
        q = isaak_adma_msg.pose_cg.orientation
        orientation = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=False)[-1]
        self.isaak_detected_object = Object(x, y, z, orientation)
        self.previous_objects = []

    def pcd_callback(self, msg):
        # 1.- convert the point cloud from ros format to numpy array --> output shape --> (N_points x 4) (x, y, z, intensity)
        points = self.get_points_from_point_cloud(msg)

        # 2.- Obtaining the transf. matrix from "/basestation" to "/vehicle_marie/velodyne" at the time stamp given from the header
        basestation_T_velodyne = self.transformer.asMatrix("/basestation", msg.header)

        # 3.- Transform the points from velodyne coordinates frame to basestation coordinates
        points = self.transform_points(points, basestation_T_velodyne)

        # 4.- Region of interest subtraction --> to remove points outside the track
        points = self.roi_removal(points)

        # 5.- ground subtraction --> to remove ground points --> choose one below
        # points = self.z_based_ground_removal(points, basestation_T_velodyne[2, 3])
        points = self.ransac_based_ground_removal(points)

        # 6.- outlier filter
        points = self.outlier_filter(points)

        # publish the processed point cloud for visualization
        header = msg.header
        header.frame_id = "basestation"
        foreground_points_as_message = self.get_point_cloud_from_points(points, header)
        self.points_publisher.publish(foreground_points_as_message)

        # 7.- clustering to generate boxes / clusters of points --> potential objects
        clusters = self.clustering(points)

        # 8.- track the clusters and create the bounding box for the CO vehicle
        bboxes = self.object_detection_naive(clusters, header)
        # bboxes = self.object_detection_standard(clusters, header)

        # publish the bounding box
        self.bbox_publisher.publish(bboxes)

        # 9.- ttc estimation
        # ttc = self.ttc_estimator()

        # publish ttc value

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
    def get_point_cloud_from_points(points, header):
        pts = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32)
        ])
        pts['x'], pts['y'], pts['z'], pts['intensity'] = points.transpose()
        pcl2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(pts, stamp=header.stamp, frame_id=header.frame_id)
        return pcl2_msg

    @staticmethod
    def transform_points(points, basestation_T_velodyne):
        all_points_in_velodyne_coordinates = np.ones_like(points)
        all_points_in_velodyne_coordinates[:, :3] = points[:, :3]
        all_points_in_global_coordinates = all_points_in_velodyne_coordinates.dot(basestation_T_velodyne.T)
        all_points_in_global_coordinates[:, -1] = points[:, -1]
        return all_points_in_global_coordinates

    def roi_removal(self, points):
        inlier_flag = self.track_path.contains_points(points[:, :2])
        return points[inlier_flag]

    @staticmethod
    def z_based_ground_removal(points, basestation_T_velodyne_z_offset):
        threshold = -1.5
        return points[points[:, 2] - basestation_T_velodyne_z_offset > threshold]

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
            # print(transformation_matrix_2d[:-1, -1])
            transformation_matrix_2d = np.linalg.inv(transformation_matrix_2d)
            box.pose.position = Point(transformation_matrix_2d[0, -1], transformation_matrix_2d[1, -1], (z_max + z_min) / 2)
            print(box.pose.position)
            rotation_matrix_3d = transformation_matrix_2d
            rotation_matrix_3d[:2, -1] = 0
            box_orientation = Rotation.from_matrix(rotation_matrix_3d).as_quat()
            box.pose.orientation = Quaternion(*box_orientation)
            box.dimensions = Vector3(length, width, z_max - z_min)
            boxes.append(box)
        # print(len(clusters))

        bboxes = BoundingBoxArray()
        bboxes.header = header
        bboxes.boxes = boxes
        return bboxes

    def object_detection_standard(self, clusters, header):
        all_distances = []
        for cluster in clusters:
            mean_point = cluster.mean(axis=0)
            distance_to_previous_detection = math.sqrt(
                (mean_point[0] - self.isaak_detected_object.locations[-1][0])**2 +
                (mean_point[1] - self.isaak_detected_object.locations[-1][1])**2
            )
            all_distances.append(distance_to_previous_detection)

        min_index = np.argmin(all_distances)

        cluster = clusters[min_index]
        z_min, z_max = cluster[:, 2].min(), cluster[:, 2].max()
        transformation_matrix_2d, (_, _) = oriented_bounds_2D(cluster[:, :2])
        transformation_matrix_2d = np.linalg.inv(transformation_matrix_2d)
        position = transformation_matrix_2d[0, -1], transformation_matrix_2d[1, -1], (z_max + z_min) / 2

        orientation = math.atan2(transformation_matrix_2d[1, 0], transformation_matrix_2d[0, 0])

        filtered_position = self.isaak_detected_object.filter_position(*position)
        filtered_orientation, filtered_speed = self.isaak_detected_object.filter_orientation_and_speed(orientation)

        print(filtered_speed)

        rotation_matrix = np.array([[math.cos(-filtered_orientation), math.sin(-filtered_orientation)],
                                    [-math.sin(-filtered_orientation), math.cos(-filtered_orientation)]])
        # print(cluster[0], filtered_position[:2])
        rotated_cluster_abs = np.abs((cluster[:, :2] - np.asarray(filtered_position[:2]).reshape(1, 2)).dot(rotation_matrix))
        length, width = np.abs(rotated_cluster_abs[:, 0]).max() * 2, np.abs(rotated_cluster_abs[:, 1]).max() * 2
        length, width = max(length, width), min(length, width)
        dimensions = length, width, z_max - z_min

        filtered_dimensions = self.isaak_detected_object.filter_dimensions(*dimensions)

        box = BoundingBox()
        box.header = header
        box.pose.position = Point(*filtered_position)
        box_orientation = Rotation.from_euler('xyz', [0, 0, filtered_orientation]).as_quat()
        box.pose.orientation = Quaternion(*box_orientation)
        box.dimensions = Vector3(*filtered_dimensions)

        bboxes = BoundingBoxArray()
        bboxes.header = header
        bboxes.boxes = [box]
        return bboxes

    def ttc_estimator(self):
        return 0

    def adma_callback_marie(self, msg):
        latest_marie_position = msg.pose_cg.position.x, msg.pose_cg.position.y, msg.pose_cg.position.z
        latest_marie_velocity = msg.velocity
        self.latest_marie_position_and_velocity = latest_marie_position, latest_marie_velocity


rospy.init_node('blas')
TTCEstimator()
rospy.spin()
