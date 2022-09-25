#!/usr/bin/env python3

import time
from pathlib import Path

import numpy as np
import pandas as pd
import rospy
from geometry_msgs.msg import PolygonStamped, Point32, Polygon

rospy.init_node("~")
track_data_pd = pd.read_csv(Path(rospy.get_param("/package_root")) / "asset" / "outdoor_contour_track_data_sp80.csv")
track_data = np.array(track_data_pd[["x_relative", "y_relative", "z_relative"]])
pub = rospy.Publisher("/outdoor_track_data", PolygonStamped, queue_size=10)
msg = PolygonStamped()
msg.polygon = Polygon([Point32(x, y, 0) for x, y, _ in track_data])
msg.header.frame_id = "basestation"

while not rospy.is_shutdown():
    pub.publish(msg)
    time.sleep(1)
