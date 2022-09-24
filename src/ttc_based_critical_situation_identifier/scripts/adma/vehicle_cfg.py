import numpy as np

isaak = dict(
    port=1210,
    color="red",
    length=2.695,  #2.83,
    width=1.752,
    height=1.542,
    dx_adma_rear_axle=0.126,
    dy_adma_rear_axle=-0.099,
    dz_adma_rear_axle=-0.737,
    rear_axle_to_gc=0.944,  # 1.09445,
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
    length=2.695,  # 2.83,
    width=1.752,
    height=1.542,
    dx_adma_rear_axle=0.109,
    dy_adma_rear_axle=-0.112,
    dz_adma_rear_axle=-0.720,
    rear_axle_to_gc=0.944,  # 1.09445,
    velodyne_port=2368,
    vehiclecg_T_velodyne=np.array([
        [0, 1, 0, -0.015],
        [-1, 0, 0, 0.009],
        [0, 0, 1, 0.8],
        [0, 0, 0, 1]
    ])
)

#marie['vehiclecg_T_velodyne'] = np.array([[ 0.04100374,  0.99915899,  0.      , 0],
#                                         [-0.99915899,  0.04100374, -0.     , 0],
#                                         [-0., 0., 1.   ,0.8],
#                                         [0, 0, 0, 1]
#]
#)

# Rotation.from_euler('xyz', [0, 0, -87.65], degrees=True).as_matrix()