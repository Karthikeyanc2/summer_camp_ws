import numpy as np

issaak = dict(
    port=1210,
    color="red",
    length=2.83,
    width=1.752,
    height=1.542,
    dx_adma_rear_axle=0.126,
    dy_adma_rear_axle=-0.099,
    dz_adma_rear_axle=-0.737,
    rear_axle_to_gc=1.29445,
    velodyne_port=2368,
    vehiclecg_T_velodyne=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
)

marie = dict(
    port=1211,
    color="blue",
    length=2.83,
    width=1.752,
    height=1.542,
    dx_adma_rear_axle=0.109,
    dy_adma_rear_axle=-0.112,
    dz_adma_rear_axle=-0.720,
    rear_axle_to_gc=1.09445,
    velodyne_port=2368,
    vehiclecg_T_velodyne=np.array([
        [0, 1, 0, 0.1],
        [-1, 0, 0, 0],
        [0, 0, 1, 0.8],
        [0, 0, 0, 1]
    ])
)