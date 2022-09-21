#!/usr/bin/env python3

import socket
import rospy
from vehicle_msgs.msg import Buffer

rospy.init_node('~')

if bool(rospy.get_param('/replay')):
    import sys
    sys.exit()

data_port = int(rospy.get_param('port_adma'))
sock_data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_data.bind(("0.0.0.0", data_port))

data_publisher = rospy.Publisher("data_raw", Buffer, queue_size=100)

data_msg = Buffer()


while not rospy.is_shutdown():
    pkt, addr = sock_data.recvfrom(99999)
    data_msg.buf = pkt
    data_publisher.publish(data_msg)
