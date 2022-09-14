#!/usr/bin/env python3

import rospy
import ros_numpy
from geometry_msgs.msg import Point, Quaternion, Vector3
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
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


class TTCEstimator:
    def __init__(self):
        self.adma_msg = None
        self.package_path = rospkg.RosPack().get_path("ttc_based_critical_situation_identifier")
        track_data = np.array(pd.read_csv(Path(self.package_path) / "scripts/outdoor_contour_track_data_sp80.csv")[["x_relative", "y_relative"]])
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(track_data, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(-2)
        self.track_path = MatplotlibPath(solution[0])
        self.transformer = tf.TransformListener(True, rospy.Duration(10))
        self.transformer.setUsingDedicatedThread(True)
        # rospy.Subscriber("/vehicle_marie/adma_data", ADMAData, self.adma_callback)
        self.ransac_regressor = linear_model.RANSACRegressor(residual_threshold=0.5, max_trials=100)
        self.points_publisher = rospy.Publisher('processed_points', PointCloud2, queue_size=10)
        self.bbox_publisher = rospy.Publisher('objects', BoundingBoxArray, queue_size=10)
        # temp_msg = rospy.wait_for_message("velodyne_points", PointCloud2, 1)
        # self.transformer.waitForTransform("/basestation", temp_msg.header.frame_id, temp_msg.header.stamp, rospy.Duration(3))
        rospy.Subscriber("velodyne_points", PointCloud2, self.pcd_callback)

    @staticmethod
    def get_points_from_point_cloud(msg):
        array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        print(array.dtype)
        points = np.zeros((len(array), 4), dtype=np.float64)
        points[..., 0] = array['x']
        points[..., 1] = array['y']
        points[..., 2] = array['z']
        points[..., 3] = array['intensity']
        return points

    @staticmethod
    def get_point_cloud_from_points(points, original_msg):
        print(points.shape)
        pts = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32)
        ])
        pts['x'], pts['y'], pts['z'], pts['intensity'] = points.transpose()
        print(pts.dtype.names)
        pcl2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(pts, stamp=original_msg.header.stamp, frame_id=original_msg.header.frame_id)
        return pcl2_msg

    def adma_callback(self, msg):
        """
        self.global_T_lidar for every 10 messages (if timestamp % 100 ms == 0)
        """
        self.adma_msg = msg

        """
        ADMA 100 Hz -->
        
        
        
        """

    def pcd_callback(self, msg):
        copy_header = msg.header
        copy_header.stamp = copy_header.stamp - rospy.Duration(0.1)
        basestation_T_velodyne = self.transformer.asMatrix("/basestation", copy_header)
        points = self.get_points_from_point_cloud(msg)
        points = self.roi_removal(points, basestation_T_velodyne)
        points = self.z_based_ground_removal(points)
        clusters = self.clustering(points)
        print([c.shape for c in clusters])
        bboxes = self.object_detection(clusters, msg.header)
        # print('start printing bboxes')
        # print('', bboxes)
        # print('stop printing bboxes')
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
        threshold = -1
        return points[points[:, 2] > threshold]

    def ransac_based_ground_removal(self, points):
        self.ransac_regressor.fit(points[:, :2], points[:, 2])
        outlier_mask = np.logical_not(self.ransac_regressor.inlier_mask_)
        return points[outlier_mask]

    @staticmethod
    def clustering(points):
        labels, core_samples_mask = DBSCAN(points[:, :2].astype(np.float64), eps=0.8, min_samples=10)
        unique_labels = set(labels)

        clusters = []
        for k in unique_labels:
            if k != -1:
                class_members_mask = labels == k
                this_cluster = points[class_members_mask & core_samples_mask]
                if len(this_cluster) > 20:
                    clusters.append(this_cluster)

        return clusters

    @staticmethod
    def object_detection(clusters, header):
        boxes = []
        for cluster in clusters:
            box = BoundingBox()
            box.header = header

            z_min, z_max = cluster[:, 2].min(), cluster[:, 2].max()
            transformation_matrix_2d, (length, width) = oriented_bounds_2D(cluster[:, :2])
            transformation_matrix_2d = np.linalg.inv(transformation_matrix_2d)
            box.pose.position = Point(transformation_matrix_2d[0, -1], transformation_matrix_2d[1, -1], (z_max + z_min) / 2)
            rotation_matrix_3d = transformation_matrix_2d
            rotation_matrix_3d[:2, -1] = 0
            box_orientation = Rotation.from_matrix(rotation_matrix_3d).as_quat()
            box.pose.orientation = Quaternion(*box_orientation)
            box.dimensions = Vector3(length, width, z_max - z_min)
            boxes.append(box)

        bboxes = BoundingBoxArray()
        bboxes.header = header
        bboxes.boxes = boxes
        return bboxes

    def ttc_estimator(self, cluster, adma_information):
        pass
        # ml inference
        # return ttc


rospy.init_node('~')
TTCEstimator()
rospy.spin()
