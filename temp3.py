#!/usr/bin/env python3
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.settimeout(1.0)
sock.bind(("0.0.0.0", 1211))

try:
    while True:
        data, _ = sock.recvfrom(856)
        print('Lateral:', data[840], "longitudinal:", data[841], "Steady state: ", data[842])
finally:
    sock.close()
