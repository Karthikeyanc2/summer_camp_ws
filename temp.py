#!/usr/bin/env python3

import socket
from struct import *

import numpy as np
import pyproj
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


lato = "48.78461825"
lono = "11.47318240"
projector = UTMProjector(lato, lono)
jan1_12am_1980 = 315532800 + 5 * 60*60*24

# create scoket and bind
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.settimeout(1.0)
sock.bind(("0.0.0.0", 1212))

try:
    while True:
        data, _ = sock.recvfrom(856)
        print(data[840], data[841], data[842])
        # print(bin(data[96])[2:].zfill(8), int(bin(data[96])[2:].zfill(8)[-4:], 2))
        # print(data)
        continue
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

finally:
    sock.close()