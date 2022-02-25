#!/usr/bin/env python3

import math
import socket
from struct import *

import numpy as np
import pyproj
import rospy
import tf2_ros
from geometry_msgs.msg import Vector3, Point, Quaternion, TransformStamped
from jsk_recognition_msgs.msg import BoundingBox
from vehicle_msgs.msg import AdmaData
from vehicle_cfg import issaak, marie
from scipy.spatial.transform import Rotation


class UTMProjector(pyproj.Proj):
    def __init__(self, lat_origin, lon_origin, alt_origin=374.565, zone=32):
        super(UTMProjector, self).__init__(proj="utm", zone=zone)
        self.lat_or = lat_origin
        self.lon_or = lon_origin
        self.alt_or = alt_origin
        self.x_or, self.y_or = self(lon_origin, lat_origin)

    def forward(self, lat, lon, alt=0.0, return_absolute=False):
        x, y = self(lon, lat)
        if return_absolute:
            return x, y, alt
        else:
            return x - self.x_or, y - self.y_or, alt - self.alt_or

    def reverse(self, x, y, z=0.0, from_absolute=False):
        if not from_absolute:
            x = x + self.x_or
            y = y + self.y_or
            lon, lat = self(x, y, inverse=True)
            return lat, lon, z + self.alt_or
        else:
            lon, lat = self(x, y, inverse=True)
            return lat, lon, z


rospy.init_node("~")

# basic parameters
lato = rospy.get_param("/lat_origin")
lono = rospy.get_param("/lon_origin")
projector = UTMProjector(lato, lono)

# adma message initialization
adma_message = AdmaData()
adma_message.obj_id = int(rospy.get_param("vehn"))
length = float(rospy.get_param("length"))
width = float(rospy.get_param("width"))
height = float(rospy.get_param("height"))
adma_message.dimensions = Vector3(length, width, height)
adma_message.header.frame_id = "basestation"
adma_message_publisher = rospy.Publisher('adma_data', AdmaData, queue_size=10)

# vehicle marker initialization
vehicle_marker = BoundingBox()
tf_prefix = str(rospy.get_param("tf_prefix"))
vehicle_marker.header.frame_id = tf_prefix + "/vehicle_cg"
vehicle_marker.pose.position = Point(0, 0, 0)
vehicle_marker.pose.orientation = Quaternion(0, 0, 0, 1)
vehicle_marker.dimensions = adma_message.dimensions
vehicle_marker_publisher = rospy.Publisher('vehicle_marker', BoundingBox, queue_size=10)

# get and set other parameters
dx_adma_rear_axle = float(rospy.get_param("dx_adma_rear_axle"))
dy_adma_rear_axle = float(rospy.get_param("dy_adma_rear_axle"))
dz_adma_rear_axle = float(rospy.get_param("dz_adma_rear_axle"))
rear_axle_to_gc = float(rospy.get_param("rear_axle_to_gc"))
vehicle_config = eval(str(rospy.get_param("vehicle_name")))
jan1_12am_1980 = 315532800 + 5 * 60*60*24  # 1980 day 1 is tuesday but as adma time always starts from sunday, we need to add 5 days to it
imuPcg = np.array([dx_adma_rear_axle + rear_axle_to_gc, dy_adma_rear_axle, dz_adma_rear_axle + height / 2, 1]).reshape(4, -1)
UDP_IP = "0.0.0.0"
UDP_PORT = int(rospy.get_param("port_adma"))

# static tf publisher for vehicle_cg and velodyne
static_transformStamped = TransformStamped()
static_transformStamped.header.stamp = rospy.Time.now()
static_transformStamped.header.frame_id = tf_prefix + "/vehicle_cg"
static_transformStamped.child_frame_id = tf_prefix + "/velodyne"
static_transformStamped.transform.translation = Vector3(*vehicle_config["vehiclecg_T_velodyne"][:-1, -1].tolist())
rotation_quaternion = Rotation.from_matrix(vehicle_config["vehiclecg_T_velodyne"][:-1, :-1]).as_quat().tolist()
static_transformStamped.transform.rotation = Quaternion(*rotation_quaternion)
tf2_ros.StaticTransformBroadcaster().sendTransform(static_transformStamped)

# tf publisher for vehicle_cg to basestation
basestation_T_vehicle_cg = TransformStamped()
basestation_T_vehicle_cg.header.stamp = rospy.Time.now()
basestation_T_vehicle_cg.header.frame_id = "basestation"
basestation_T_vehicle_cg.child_frame_id = tf_prefix + "/vehicle_cg"
basestation_T_vehicle_cg_publisher = tf2_ros.TransformBroadcaster()

# create scoket and bind
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.settimeout(1.0)
sock.bind((UDP_IP, UDP_PORT))

while not rospy.is_shutdown():
    data, _ = sock.recvfrom(856)

    # get time
    time_ms = unpack('I', data[584:588])[0]
    time_week = unpack('H', data[588:590])[0]
    time_adma = time_ms * 0.001 + time_week * 7 * 24 * 60 * 60 + jan1_12am_1980

    # get pose
    orientation = (unpack('H', data[508:510])[0] * 0.01 + 90.0) * np.pi / 180
    # INS absolute
    lat = unpack('i', data[592:596])[0] * 1e-7
    lon = unpack('i', data[596:600])[0] * 1e-7
    x, y, _ = projector.forward(lat, lon)
    # INS relative
    # y = unpack('i', data[600:604])[0] * 0.01
    # x = unpack('i', data[604:608])[0] * 0.01
    # lat, lon, _ = projector.reverse(x, y)
    # INS absolute POI1
    # lat = unpack('i', data[608:612])[0] * 1e-7
    # lon = unpack('i', data[612:616])[0] * 1e-7
    # x, y, _ = projector.forward(lat, lon)
    # INS relative POI1
    # y = unpack('i', data[616:620])[0] * 0.01
    # x = unpack('i', data[620:624])[0] * 0.01
    # lat, lon, alt = projector.reverse(x, y)
    basestationPimu = np.array([x, y, -dz_adma_rear_axle, 1]).reshape(4, -1)
    basestationTimu = np.identity(4)
    basestationTimu[:2, :2] = np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    basestationTimu[:, -1] = basestationPimu.reshape(-1)
    basestationPcg = basestationTimu.dot(imuPcg)  # if you want rear center you need to change it here

    # get velocity
    vx = unpack('h', data[736:738])[0] * 0.005
    vy = unpack('h', data[738:740])[0] * 0.005

    # get acceleration
    ax = unpack('h', data[160:162])[0] * 0.0004 * 9.81
    ay = unpack('h', data[162:164])[0] * 0.0004 * 9.81

    # populate and publish adma data
    adma_message.header.stamp = rospy.Time.from_sec(time_adma)
    rotation_quaternions = Rotation.from_euler('xyz', [0, 0, orientation]).as_quat().tolist()
    adma_message.pose_cg.orientation = Quaternion(*rotation_quaternions)
    adma_message.pose_cg.position = Point(*basestationPcg.reshape(-1).tolist()[:-1])
    adma_message.velocity = math.sqrt(vx ** 2 + vy ** 2)
    adma_message.acceleration = ax  # np.sign(ax) * math.sqrt(ax ** 2 + ay ** 2)
    adma_message_publisher.publish(adma_message)

    # publish vehicle marker
    vehicle_marker_publisher.publish(vehicle_marker)

    # publish tf between vehicle_cg and basestation
    basestation_T_vehicle_cg.header.stamp = rospy.Time.now()
    basestation_T_vehicle_cg.transform.translation = adma_message.pose_cg.position
    basestation_T_vehicle_cg.transform.rotation = adma_message.pose_cg.orientation
    basestation_T_vehicle_cg_publisher.sendTransform(basestation_T_vehicle_cg)
