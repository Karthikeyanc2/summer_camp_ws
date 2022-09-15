#!/usr/bin/env python3

import select
import socket
import rospy
from vehicle_msgs.msg import Buffer

rospy.init_node('~')

data_port = int(rospy.get_param('data_port'))
sock_data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_data.bind(("0.0.0.0", data_port))

tele_port = int(rospy.get_param('tele_port'))
sock_tele = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_tele.bind(("0.0.0.0", tele_port))

data_publisher = rospy.Publisher("data_raw", Buffer, queue_size=100)
tele_publisher = rospy.Publisher("tele_raw", Buffer, queue_size=100)

data_msg = Buffer()
tele_msg = Buffer()


while not rospy.is_shutdown():
    ready_read, _, _ = select.select([sock_data, sock_tele], [], [], None)
    for ready in ready_read:
        pkt, addr = ready.recvfrom(99999)
        if addr[1] == data_port:
            data_msg.buf = pkt
            data_publisher.publish(data_msg)
        else:
            tele_msg.buf = pkt
            tele_publisher.publish(tele_msg)
