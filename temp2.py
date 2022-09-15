import socket
import struct
import select
import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# """
UDP_IP = "0.0.0.0"
UDP_PORT = 2368

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(99999)
    t = round(struct.unpack('I', data[-6:-2])[0], 2) * 0.000001
    print("received message: " ,  len(data), t, struct.unpack('cc', data[-2:]))
    print(t - time.time()%3600)
# """

"""

UDP_IP = "0.0.0.0"
UDP_PORT = 8308

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(99999)
    print("received message: ",  len(data), data.find(b'GPRMC'), data[206:278])
"""

"""
sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock1.bind(("0.0.0.0", 2368))

sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2.bind(("0.0.0.0", 8308))

while True:
     # data, addr = sock.recvfrom(99999)
     ready_read, _, _ = select.select([sock1, sock2], [], [], None)
     for ready in ready_read:
         pkt, addr = ready.recvfrom(99999)
         print(addr, len(pkt))
"""

ele_angles = np.broadcast_to(np.expand_dims(np.array(
    [-0.535293, -0.511905, -0.488692, -0.465479, -0.442092, -0.418879, -0.395666, -0.372279, -0.349066, -0.325853,
     -0.302466, -0.279253, -0.256040, -0.232652, -0.209440, -0.186227, -0.162839, -0.139626, -0.116413, -0.093026,
     -0.069813, -0.046600, -0.023213, +0.000000, +0.023213, +0.046600, +0.069813, +0.093026, +0.116413, +0.139626,
     +0.162839, +0.186227]  # [::-1]
), axis=0), (12, 32))  # 12 x 32
cos_ele_angles = np.cos(ele_angles)
sin_ele_angles = np.sin(ele_angles)

sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock1.bind(("0.0.0.0", 2368))

sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2.bind(("0.0.0.0", 8308))

lidar_packet_format = "<" + ("HH" + "HB" * 32) * 12 + "Ixx"
unpack_lidar = struct.Struct(lidar_packet_format).unpack

all_points = np.array([]).reshape(-1, 6)
c = 0
previous_utc_time_from_lidar = 0
while True:
    ready_read, _, _ = select.select([sock1, sock2], [], [], None)
    for ready in ready_read:
        pkt, addr = ready.recvfrom(99999)
        if addr[1] == 2368:
            unpacked = unpack_lidar(pkt)
            utc_time_from_lidar = unpacked[-1] * 1 ^ -6
            array = np.asarray(unpacked[:-1]).reshape(12, 66)  # 12 x 32 points
            azi_angles = array[:, 1].reshape(-1, 1) * np.pi / 18000.0
            cos_azi_angles = np.cos(azi_angles)
            sin_azi_angles = np.sin(azi_angles)
            distances = array[:, 2::2] / 2000.0
            xs = distances * sin_azi_angles * cos_ele_angles
            ys = distances * cos_azi_angles * cos_ele_angles
            zs = distances * sin_ele_angles
            intensities = array[:, 3::2]
            all_points = np.r_[all_points, np.c_[xs.reshape(-1, 1), ys.reshape(-1, 1), zs.reshape(-1, 1), intensities.reshape(-1, 1), distances.reshape(-1, 1)]]
            c += 1

        print(addr, len(pkt))
    if c > 170:
        break

print(all_points.shape)
fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
plt.gca().scatter(all_points[:, 0], all_points[:, 1], marker='.', s=2)
plt.gca().set_aspect(1)

plt.show()
