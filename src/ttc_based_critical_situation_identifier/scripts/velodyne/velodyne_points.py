#!/usr/bin/env python3

import struct
import time

import numpy as np
import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2
from vehicle_msgs.msg import Buffer
import datetime
import calendar

jan1_12am_1980 = 315532800


class Velodyne:
    def __init__(self):
        self.publisher = rospy.Publisher("points", PointCloud2, queue_size=10)
        ele_angles = np.expand_dims(np.array(
            [-0.535293, -0.511905, -0.488692, -0.465479, -0.442092, -0.418879, -0.395666, -0.372279, -0.349066,
             -0.325853, -0.302466, -0.279253, -0.256040, -0.232652, -0.209440, -0.186227, -0.162839, -0.139626,
             -0.116413, -0.093026, -0.069813, -0.046600, -0.023213, +0.000000, +0.023213, +0.046600, +0.069813,
             +0.093026, +0.116413, +0.139626, +0.162839, +0.186227]  # [::-1]
        ).reshape(2, -1).T.reshape(-1), axis=0)
        self.cos_ele_angles = np.cos(ele_angles)
        self.sin_ele_angles = np.sin(ele_angles)
        lidar_packet_format = "<" + ("HH" + "HB" * 32) * 12 + "Ixx"
        self.unpack_lidar = struct.Struct(lidar_packet_format).unpack

        tele_msg = str(bytes(rospy.wait_for_message("tele_raw", Buffer, 10.0).buf)[206:278])
        d = tele_msg.split(',')[9]
        t = tele_msg.split(',')[1]
        self.first_hour_utc = calendar.timegm(datetime.datetime(int('20' + d[-2:]), int(d[-4:-2]), int(d[:2]), int(t[:2]), 0, 0).timetuple())
        # t = time.time()
        # self.last_hour_utc = t - t % 3600
        # self.last_utc = 0
        self.previous_lidar_utc_time = 0

        self.tf_prefix = rospy.get_param('tf_prefix')

        self.all_array = np.array([]).reshape(-1, 66)

        rospy.Subscriber("data_raw", Buffer, callback=self.data_raw_callback)
        # rospy.Subscriber("tele_raw", Buffer, callback=self.tele_raw_callback)

    def data_raw_callback(self, msg):
        data = bytes(msg.buf)
        unpacked = self.unpack_lidar(data)

        array = np.asarray(unpacked[:-1]).reshape(12, 66)  # 12 x 32 points
        self.all_array = np.r_[self.all_array, array]

        all_azi_angles = self.all_array[:, 1]
        zero_crossing_index = np.where(np.diff(all_azi_angles) < 0)[0]

        if zero_crossing_index.size:
            utc_time_from_lidar = unpacked[-1] * 0.000001 + self.first_hour_utc
            if self.previous_lidar_utc_time > utc_time_from_lidar:
                utc_time_from_lidar += 3600
            self.previous_lidar_utc_time = utc_time_from_lidar
            this_array = self.all_array[:zero_crossing_index[0] + 1]
            self.all_array = self.all_array[zero_crossing_index[0] + 1:]

            azi_angles = np.expand_dims(this_array[:, 1] * np.pi / 18000.0, axis=1)
            cos_azi_angles = np.cos(azi_angles)
            sin_azi_angles = np.sin(azi_angles)
            distances = this_array[:, 2::2] / 500.0
            xs = distances * cos_azi_angles * self.cos_ele_angles
            ys = - distances * sin_azi_angles * self.cos_ele_angles
            zs = distances * self.sin_ele_angles
            intensities = this_array[:, 3::2]
            # print(xs.view(1, -1), ys.view(1, -1), zs.view(1, -1), intensities.view(1, -1))
            points = np.c_[xs.reshape(-1, 1), ys.reshape(-1, 1), zs.reshape(-1, 1),
                           intensities.reshape(-1, 1)][distances.reshape(-1) > 1]

            pts = np.zeros(len(points), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('intensity', np.float32)
            ])
            pts['x'], pts['y'], pts['z'], pts['intensity'] = points.transpose()
            pcl2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(pts,
                                                                   stamp=rospy.Time.from_seconds(utc_time_from_lidar - 0.1),  # rospy.get_rostime(),
                                                                   frame_id=self.tf_prefix + "/velodyne")
            self.publisher.publish(pcl2_msg)
            # print(round(time.time() - utc_time_from_lidar, 3), round(utc_time_from_lidar, 3))

    def tele_raw_callback(self, msg):
        print(bytes(msg.buf)[206:278])


rospy.init_node("bla")
Velodyne()
rospy.spin()